import base64
import json
import os
import re

import tenseal as ts
from cryptography.fernet import Fernet

# Path to store or load the AES symmetric encryption key for payload security
AES_KEY_PATH = "secret.key"


def parse_dnf_query(query_str: str, vocabulary: list):
    """
    Parses a Boolean search query string into Disjunctive Normal Form (DNF) clauses.

    Maps tokens to the given vocabulary, tracks out-of-vocabulary (OOV) terms,
    and returns formatted clause structures along with ignored terms.
    """
    if not query_str or not query_str.strip():
        return [], []

    # Map lowercased vocabulary terms to their corresponding index for fast lookup
    vocab_lower_map = {term.lower(): idx for idx, term in enumerate(vocabulary)}

    # Extract operators, structural parentheses, and alphanumeric word tokens
    tokens = re.findall(r"\(|\)|\bAND\b|\bOR\b|\bNOT\b|\w+", query_str, re.IGNORECASE)

    def tokenize_expr(toks):
        """Recursively parses query tokens using standard operator precedence (NOT > AND > OR)."""

        def parse_primary(i):
            if i >= len(toks):
                return [], i
            tok = toks[i]

            # Handle grouped expressions within parentheses
            if tok == "(":
                res, i = parse_or(i + 1)
                if i < len(toks) and toks[i] == ")":
                    i += 1
                return res, i

            # Handle logical NOT operator (applies De Morgan's laws to negate inner terms)
            elif tok.upper() == "NOT":
                sub, i = parse_primary(i + 1)
                negated_dnf = []
                for clause in sub:
                    new_clause = []
                    for item in clause:
                        if "pos" in item:
                            new_clause.append({"neg": item["pos"]})
                        elif "neg" in item:
                            new_clause.append({"pos": item["neg"]})
                    negated_dnf.append(new_clause)
                return negated_dnf, i

            # Base term expression
            else:
                term_lower = tok.lower()
                i += 1
                return [[{"pos": term_lower}]], i

        def parse_and(i):
            """Parses AND operator expressions (conjunctions within clauses)."""
            left, i = parse_primary(i)
            while i < len(toks) and toks[i].upper() == "AND":
                right, i = parse_primary(i + 1)
                new_dnf = []
                for c1 in left:
                    for c2 in right:
                        new_dnf.append(c1 + c2)
                left = new_dnf
            return left, i

        def parse_or(i):
            """Parses OR operator expressions (disjunctions of clauses)."""
            left, i = parse_and(i)
            while i < len(toks) and toks[i].upper() == "OR":
                right, i = parse_and(i + 1)
                left = left + right
            return left, i

        dnf, _ = parse_or(0)
        return dnf

    raw_dnf = tokenize_expr(tokens)

    formatted_clauses = []
    ignored_terms = set()

    # Convert raw DNF clauses to integer indices against the provided vocabulary
    for clause in raw_dnf:
        pos_indices = []
        neg_indices = []
        for item in clause:
            if "pos" in item:
                w = item["pos"].lower()
                if w in vocab_lower_map:
                    pos_indices.append(vocab_lower_map[w])
                else:
                    ignored_terms.add(w)
            elif "neg" in item:
                w = item["neg"].lower()
                if w in vocab_lower_map:
                    neg_indices.append(vocab_lower_map[w])
                else:
                    ignored_terms.add(w)

        # Retain clauses containing at least one valid vocabulary term
        if pos_indices or neg_indices:
            formatted_clauses.append(
                {
                    "pos_indices": list(set(pos_indices)),
                    "neg_indices": list(set(neg_indices)),
                }
            )

    return formatted_clauses, list(ignored_terms)


class FHEClient:
    """
    Client class managing Fully Homomorphic Encryption (FHE) operations via TenSEAL (BFV scheme)
    and traditional symmetric encryption via Fernet (AES-128).
    """

    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

        # Initialize BFV FHE context with standard security parameter setup
        self.context = ts.context(
            ts.SCHEME_TYPE.BFV, poly_modulus_degree=8192, plain_modulus=1032193
        )
        self.context.generate_relin_keys()
        self.context.generate_galois_keys()

        # Load existing symmetric key or create a new one if not present
        if os.path.exists(AES_KEY_PATH):
            with open(AES_KEY_PATH, "rb") as f:
                self.aes_key = f.read()
        else:
            self.aes_key = Fernet.generate_key()
            with open(AES_KEY_PATH, "wb") as f:
                f.write(self.aes_key)

        self.fernet = Fernet(self.aes_key)

    def get_public_context_bytes(self) -> bytes:
        """Generates a serialized public FHE context (stripped of the private key) for server transmission."""
        server_context = self.context.copy()
        server_context.make_context_public(
            generate_galois_keys=True, generate_relin_keys=True
        )
        return server_context.serialize(save_secret_key=False)

    def encrypt_dnf_vectors(self, query_str: str, vocabulary: list):
        """
        Parses a query into DNF form and encrypts positive indicator term vectors using FHE.

        Returns Base64-encoded encrypted clause data and a list of ignored/unmatched terms.
        """
        clauses, ignored_terms = parse_dnf_query(query_str, vocabulary)
        encrypted_clauses = []

        for clause in clauses:
            pos_vec = [0] * self.vocab_size

            if clause["pos_indices"]:
                for idx in clause["pos_indices"]:
                    pos_vec[idx] = 1
            else:
                # Set dummy anchor index for pure-negation clauses
                pos_vec[0] = 1

            # Homomorphically encrypt the positive terms boolean vector
            enc_pos = ts.bfv_vector(self.context, pos_vec)

            encrypted_clauses.append(
                {
                    "enc_pos_b64": base64.b64encode(enc_pos.serialize()).decode(
                        "utf-8"
                    ),
                    "required_pos_count": len(clause["pos_indices"]),
                    "is_pure_negation": len(clause["pos_indices"]) == 0,
                    "neg_indices": clause["neg_indices"],
                }
            )

        return encrypted_clauses, ignored_terms

    def decrypt_results(self, serialized_ciphers: list) -> list:
        """Decrypts a batch of serialized BFV ciphertexts into integer values."""
        results = []
        for cipher_bytes in serialized_ciphers:
            cipher = ts.bfv_vector_from(self.context, cipher_bytes)
            results.append(int(cipher.decrypt()[0]))
        return results

    def decrypt_dnf_results(self, dnf_results_payload: list, num_docs: int) -> list:
        """
        Decrypts FHE search scores returned by the server and evaluates document relevance matches.

        Evaluates matches according to clause requirements (positive requirement counts and pure-negation flags).
        """
        # If query is empty/unrestricted, default to matching all documents
        if not dnf_results_payload:
            return [1] * num_docs

        final_matches = [0] * num_docs

        for clause_res in dnf_results_payload:
            req_count = clause_res["required_pos_count"]
            is_pure_neg = clause_res.get("is_pure_negation", False)
            serialized_scores = clause_res["encrypted_scores"]

            # Decrypt homomorphic scores per document for this clause
            decrypted_scores = self.decrypt_results(
                [base64.b64decode(s) for s in serialized_scores]
            )

            for doc_idx in range(num_docs):
                score = decrypted_scores[doc_idx]

                if is_pure_neg:
                    # Pure negation match requirement check
                    if score == 1:
                        final_matches[doc_idx] = 1
                else:
                    # Positive clause threshold evaluation
                    if req_count > 0 and score >= req_count:
                        final_matches[doc_idx] = 1

        return final_matches

    def try_decrypt_single(self, cipher_bytes: bytes):
        """Helper to deserialize and decrypt a single raw BFV ciphertext vector."""
        cipher = ts.bfv_vector_from(self.context, cipher_bytes)
        return cipher.decrypt()

    def encrypt_payload(self, text: str) -> str:
        """Encrypts a plaintext string using Fernet (AES-128) symmetric encryption."""
        return self.fernet.encrypt(text.encode("utf-8")).decode("utf-8")

    def decrypt_payload(self, cipher_str: str) -> str:
        """Decrypts a Fernet (AES-128) ciphertext string back to plaintext."""
        return self.fernet.decrypt(cipher_str.encode("utf-8")).decode("utf-8")

    def encrypt_benchmark_data(self, data_dict: dict) -> str:
        """Serializes a dictionary to JSON and encrypts it as a Base64-encoded Fernet payload."""
        json_bytes = json.dumps(data_dict).encode("utf-8")
        encrypted_bytes = self.fernet.encrypt(json_bytes)
        return base64.b64encode(encrypted_bytes).decode("utf-8")

    def decrypt_benchmark_data(self, cipher_b64: str) -> dict:
        """Decrypts a Base64-encoded Fernet payload and deserializes it back into a dictionary."""
        encrypted_bytes = base64.b64decode(cipher_b64)
        decrypted_bytes = self.fernet.decrypt(encrypted_bytes)
        return json.loads(decrypted_bytes.decode("utf-8"))
