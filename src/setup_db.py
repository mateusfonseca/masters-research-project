import gc
import json
import re
import sys
from collections import Counter
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import scipy.sparse as sp
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from src.client import FHEClient

# Directory path setup pointing to project root and data directory
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Download required NLTK resource packages quietly
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# Configuration constants for dataset processing limits
MAX_DOCS = 1000
VOCAB_SIZE = 2048

print(f"1. Sampling {MAX_DOCS} emails...")
try:
    # Load and randomly sample a subset of raw emails from CSV
    df = (
        pd.read_csv(DATA_DIR / "emails.csv", usecols=["message"])
        .sample(n=MAX_DOCS, random_state=42)
        .reset_index(drop=True)
    )
except FileNotFoundError:
    print("Error: emails.csv not found in working directory.")
    sys.exit()


def extract_body(raw_message):
    """Strips MIME headers from raw email string and extracts the message body content."""
    raw_str = str(raw_message)
    clean_text = re.sub(
        r"^(Message-ID|Date|From|To|Subject|Cc|Bcc|Mime-Version|Content-Type|"
        r"Content-Transfer-Encoding|X-From|X-To|X-cc|X-bcc|X-Folder|X-Origin|"
        r"X-FileName):.*?\n",
        "",
        raw_str,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    parts = clean_text.split("\n\n", 1)
    return parts[1].strip() if len(parts) > 1 else clean_text.strip()


def extract_keywords(text):
    """Tokenizes text, filters non-alphanumeric characters, and removes stop words."""
    # Keep alphanumeric tokens
    clean_text = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    tokens = word_tokenize(clean_text)
    stop_words = set(stopwords.words("english"))
    return [w for w in tokens if w not in stop_words and len(w) > 2]


print("Extracting bodies and keywords...")
# Extract email body, keywords, and subject line into new dataframe columns
df["body"] = df["message"].apply(extract_body)
df["keywords"] = df["body"].apply(extract_keywords)
df["subject"] = df["message"].apply(
    lambda m: (
        m.split("\n")[0].replace("Subject:", "").strip()
        if "Subject:" in m
        else "No Subject"
    )
)

# Free up memory by dropping raw message column
df.drop(columns=["message"], inplace=True)
gc.collect()

print("2. Building sparse term-occurrence matrix...")
# Count document frequency for each word token
all_words = [w for sublist in df["keywords"] for w in set(sublist)]
word_counts = Counter(all_words)

# Top 2048 standard terms
top_words = [word for word, _ in word_counts.most_common(VOCAB_SIZE)]

# Filter for numeric strings with at least 4 digits
numeric_counts = Counter(
    {k: v for k, v in word_counts.items() if k.isdigit() and len(k) >= 4}
)
top_numeric = [word for word, _ in numeric_counts.most_common(1000)]

# Combine while keeping total vector length < 8192
top_vocab = list(dict.fromkeys(top_words + top_numeric))

print(
    f"Total Vocabulary Size: {len(top_vocab)} (Standard: {len(top_words)}, Numeric >= 4 digits: {len(top_numeric)})"
)

# Create index mapping dictionary for quick vocabulary lookup
word_to_idx = {word: i for i, word in enumerate(top_vocab)}

# Build sparse binary term-document matrix index arrays
rows, cols = [], []
for doc_idx, doc_words in enumerate(df["keywords"]):
    unique_words = set(doc_words)
    for word in unique_words:
        if word in word_to_idx:
            rows.append(word_to_idx[word])
            cols.append(doc_idx)

# Construct CSR sparse binary occurrence matrix
occ_matrix = sp.csr_matrix(
    (np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=(len(top_vocab), len(df))
)

# Save sparse occurrence matrix and vocabulary mapping to disk
sp.save_npz(DATA_DIR / "occurrence_matrix_sparse.npz", occ_matrix)
print(f"Sparse matrix saved! Shape: {occ_matrix.shape}")

with open(DATA_DIR / "vocabulary.json", "w") as f:
    json.dump(top_vocab, f)

print("3. Encrypting email payloads with Fernet AES...")
client = FHEClient(len(top_vocab))

# Encrypt each email payload and write out streamingly to JSON database
with open(DATA_DIR / "encrypted_emails_db.json", "w") as f:
    f.write("{\n")
    for position_idx, (_, row) in enumerate(df.iterrows()):
        raw_text = f"Subject: {row['subject']}\n\n{row['body']}"
        enc_payload = client.encrypt_payload(raw_text)

        if position_idx > 0:
            f.write(",\n")
        f.write(f'"{position_idx}": {json.dumps(enc_payload)}')

        del enc_payload
        if position_idx % 200 == 0:
            gc.collect()

    f.write("\n}")

print("Database rebuilt and indices synchronized successfully!")
