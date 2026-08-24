import argparse
import base64
import json
import logging
import os
import re
import sys
import threading
import time

import matplotlib.pyplot as plt
import pandas as pd
import psutil
import requests
import streamlit as st
import urllib3
import uvicorn
from dotenv import load_dotenv
from scipy import stats

from src.client import FHEClient

# Load environment variables from a local .env file
load_dotenv()

# Disable insecure HTTPS request warnings for self-signed certificates or IP endpoints
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Module-scoped logger for non-root reporting
logger = logging.getLogger(__name__)

# 1. PARSE COMMAND LINE ARGUMENTS & SESSION STATE OVERRIDES
# Set up command-line arguments to toggle local execution mode
parser = argparse.ArgumentParser(description="MSc Data App - FHE Search")
parser.add_argument(
    "--local",
    type=str,
    default="no",
    choices=["yes", "no"],
    help="Run the backend FastAPI server locally (yes/no)",
)
args, _ = parser.parse_known_args()

# Detect execution context (PyTest testing vs Streamlit active runtime)
IS_PYTEST = "pytest" in sys.modules or any("pytest" in arg for arg in sys.argv)
IS_RUNTIME = st.runtime.exists() and not IS_PYTEST

if IS_RUNTIME:
    # Streamlit page layout setup
    st.set_page_config(page_title="MSc Data App - FHE Search", layout="wide")

    # Persist local execution mode choice across Streamlit app reruns
    if "is_local_mode" not in st.session_state:
        st.session_state["is_local_mode"] = args.local.lower() == "yes"

    IS_LOCAL = st.session_state["is_local_mode"]
else:
    IS_LOCAL = args.local.lower() == "yes"

# 3. CONFIGURE SERVER URL
if IS_LOCAL:
    if IS_RUNTIME:
        # Spawn backend FastAPI server in a background thread if running locally
        def run_local_server():
            from src.server import app as fastapi_app

            uvicorn.run(fastapi_app, host="127.0.0.1", port=8000, log_level="debug")

        if "local_server_started" not in st.session_state:
            server_thread = threading.Thread(target=run_local_server, daemon=True)
            server_thread.start()
            st.session_state["local_server_started"] = True
            time.sleep(1)

    SERVER_URL = "http://127.0.0.1:8000"
else:
    domain = os.getenv("DUCKDNS_DOMAIN")
    if not domain:
        raise ValueError("DUCKDNS_DOMAIN is not set in the environment or .env file.")

    SERVER_URL = f"https://{domain}" if not domain.startswith("http") else domain


# 4. CORE TESTABLE HELPER FUNCTIONS


def highlight_text(text, valid_terms):
    """Highlights active search terms evaluated in the FHE matrix using HTML tags."""
    if not text or not valid_terms:
        return text

    pattern = re.compile(
        r"\b(" + "|".join(map(re.escape, valid_terms)) + r")\b", re.IGNORECASE
    )
    return pattern.sub(
        lambda match: (
            f"<mark style='background-color: #ffe066; padding: 2px 4px; border-radius: 3px; color: black; font-weight: bold;'>{match.group(0)}</mark>"
        ),
        text,
    )


def convert_df_to_csv(df):
    """Utility to convert a pandas DataFrame to CSV bytes for downloads."""
    return df.to_csv(index=False).encode("utf-8")


def preprocess_query_string(query_str):
    """Normalizes Boolean operators in the user search query to uppercase."""
    if not query_str:
        return ""
    query_str = query_str.strip()
    return re.sub(
        r"\b(and|or|not)\b",
        lambda m: m.group(0).upper(),
        query_str,
        flags=re.IGNORECASE,
    )


def initialize_remote_client(api_url):
    """
    Pings the backend server to confirm availability, fetches vocabulary,
    and initializes an instance of FHEClient.
    """
    max_retries = 10 if IS_LOCAL else 1
    server_alive = False

    # Ping backend server to verify availability
    for i in range(max_retries):
        try:
            res = requests.get(f"{api_url}/", timeout=10, verify=False)
            if res.status_code == 200:
                server_alive = True
                break
        except requests.exceptions.RequestException:
            if IS_LOCAL and i < max_retries - 1:
                time.sleep(0.5)

    # Fallback to local mode if the remote server is unreachable
    if not server_alive and not IS_LOCAL and IS_RUNTIME:
        st.warning(
            "Remote server unreachable after 10s ping. Switching to local mode and reloading..."
        )
        time.sleep(1.5)
        st.session_state["is_local_mode"] = True
        st.rerun()

    # Retrieve global vocabulary list from server
    global_vocab = None
    try:
        res = requests.get(f"{api_url}/get_vocabulary", timeout=600, verify=False)
        if res.status_code == 200:
            global_vocab = res.json()["vocabulary"]
    except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError) as e:
        logger.error(f"Vocabulary retrieval failed from {api_url}: {e}")
        if IS_RUNTIME:
            st.error(f"Failed to retrieve vocabulary from {api_url}: {e}")

    if not global_vocab:
        if IS_RUNTIME:
            st.error(f"Could not load vocabulary from backend server ({api_url}).")
            st.stop()
        return [], None

    fhe_client = FHEClient(len(global_vocab))
    return global_vocab, fhe_client


def load_email_metadata(api_url):
    """Loads email document metadata from the server into a DataFrame."""
    try:
        res = requests.get(f"{api_url}/get_email_metadata", timeout=600, verify=False)
        if res.status_code == 200:
            data = res.json()["metadata"]
            return pd.DataFrame(data)
    except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Could not load metadata from {api_url}: {e}")
        if IS_RUNTIME:
            st.warning(
                f"Could not load remote metadata from {api_url}. Server may be starting."
            )

    # Fallback placeholder DataFrame if network request fails
    return pd.DataFrame(
        [
            {
                "doc_id": i,
                "file": f"email_{i}.txt",
                "subject": f"Encrypted Email #{i}",
                "body": "[Encrypted Payload]",
            }
            for i in range(50)
        ]
    )


def load_persisted_benchmarks_from_server(api_url, fhe_client):
    """Fetches encrypted benchmark blobs from the server and decrypts them into state data."""
    benchmark_mapping = [
        ("h1_latest", "h1"),
        ("h2_latest", "h2"),
        ("h3_latest", "h3"),
    ]
    results = {}
    for b_id, data_key in benchmark_mapping:
        try:
            res = requests.get(
                f"{api_url}/get_encrypted_benchmark/{b_id}", timeout=600, verify=False
            )
            if res.status_code == 200:
                enc_blob = res.json()["encrypted_blob"]
                decrypted_data = fhe_client.decrypt_benchmark_data(enc_blob)
                if data_key in decrypted_data:
                    results[f"{data_key}_metrics"] = pd.DataFrame(
                        decrypted_data[data_key]
                    )
        except (
            requests.exceptions.RequestException,
            json.JSONDecodeError,
            KeyError,
            ValueError,
        ) as e:
            logger.info(f"Skipping benchmark restoration for '{b_id}': {e}")
    return results


def execute_fhe_search(query_str, target_doc_count, fhe_client, global_vocab, api_url):
    """Headless logic for running Homomorphic Vector DNF Search."""
    clean_query = preprocess_query_string(query_str)

    # Return match-all default state for empty queries
    if not clean_query:
        return {
            "query": "[All Messages]",
            "enc_time_ms": 0.0,
            "eval_time_ms": 0.0,
            "ram_usage_mb": 0.0,
            "cipher_size_bytes": 0,
            "matches": [1] * target_doc_count,
            "ignored_terms": [],
            "valid_terms": [],
        }, None

    pub_context_bytes = fhe_client.get_public_context_bytes()
    pub_ctx_b64 = base64.b64encode(pub_context_bytes).decode("utf-8")
    t0 = time.perf_counter()

    # Encrypt query clauses into BFV vectors
    dnf_clauses, ignored_terms = fhe_client.encrypt_dnf_vectors(
        clean_query, global_vocab
    )
    t_enc = (time.perf_counter() - t0) * 1000

    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / (1024 * 1024)
    cipher_bytes = sum(len(str(clause)) for clause in dnf_clauses)

    # Extract valid search terms excluding operators and out-of-vocabulary words
    raw_words = set(re.findall(r"\b(?i:(?!AND\b|OR\b|NOT\b))\w+\b", clean_query))
    valid_terms = [w for w in raw_words if w.lower() not in ignored_terms]

    payload = {
        "public_context_b64": pub_ctx_b64,
        "dnf_clauses": dnf_clauses,
        "target_doc_count": target_doc_count,
    }

    try:
        # Dispatch search request to server endpoint
        res = requests.post(
            f"{api_url}/fhe_boolean_search", json=payload, timeout=600, verify=False
        )
        if res.status_code == 200:
            data = res.json()
            matches = fhe_client.decrypt_dnf_results(
                data["dnf_results"], target_doc_count
            )

            return {
                "query": clean_query,
                "mode": "VECTOR_DNF",
                "enc_time_ms": t_enc,
                "eval_time_ms": data["server_processing_time_ms"],
                "ram_usage_mb": ram_mb,
                "cipher_size_bytes": cipher_bytes,
                "matches": matches,
                "ignored_terms": ignored_terms,
                "valid_terms": valid_terms,
            }, None
        else:
            return {
                "query": clean_query,
                "mode": "VECTOR_DNF",
                "enc_time_ms": t_enc,
                "eval_time_ms": 0.0,
                "ram_usage_mb": ram_mb,
                "cipher_size_bytes": 0,
                "matches": [0] * target_doc_count,
                "ignored_terms": [],
                "valid_terms": [],
            }, f"Server Error {res.status_code}: {res.text}"
    except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError) as e:
        logger.error(f"FHE search failed against {api_url}: {e}")
        return {
            "query": clean_query,
            "mode": "VECTOR_DNF",
            "enc_time_ms": 0.0,
            "eval_time_ms": 0.0,
            "ram_usage_mb": 0.0,
            "cipher_size_bytes": 0,
            "matches": [0] * target_doc_count,
            "ignored_terms": [],
            "valid_terms": [],
        }, f"Failed to connect to server at {api_url}: {e!s}"


def execute_h1_benchmark_step(
    target_max, fhe_client, global_vocab, api_url, progress_callback=None
):
    """Executes H1 overhead benchmark across dynamic dataset sizes."""
    pub_bytes = fhe_client.get_public_context_bytes()
    pub_b64 = base64.b64encode(pub_bytes).decode("utf-8")
    sample_term = global_vocab[0] if global_vocab else "test"
    dnf_clauses, _ = fhe_client.encrypt_dnf_vectors(sample_term, global_vocab)

    step_size = max(1, target_max // 5)
    sizes = list(range(step_size, target_max + 1, step_size))
    raw_metrics = []

    # Iteratively run benchmarks across dataset size increments
    for idx, size in enumerate(sizes):
        if progress_callback:
            progress_callback(idx, len(sizes), size, target_max)

        payload = {
            "public_context_b64": pub_b64,
            "dnf_clauses": dnf_clauses,
            "dataset_sizes": [size],
        }

        res = requests.post(
            f"{api_url}/run_h1_benchmark", json=payload, timeout=600, verify=False
        )
        if res.status_code == 200:
            chunk_metrics = res.json()["metrics"]
            raw_metrics.extend(chunk_metrics)
        else:
            raise RuntimeError(f"Error for size N={size}: HTTP {res.status_code}")

    # Persist benchmark result payload to server storage
    if raw_metrics:
        enc_blob = fhe_client.encrypt_benchmark_data({"h1": raw_metrics})
        requests.post(
            f"{api_url}/save_encrypted_benchmark",
            json={"benchmark_id": "h1_latest", "encrypted_blob": enc_blob},
            verify=False,
        )

    return pd.DataFrame(raw_metrics)


def execute_h2_noise_test(fhe_client, api_url):
    """Executes H2 Multi-term depth noise exhaustion test."""
    pub_bytes = fhe_client.get_public_context_bytes()
    pub_b64 = base64.b64encode(pub_bytes).decode("utf-8")

    payload = {"public_context_b64": pub_b64, "num_clause_terms": 6}

    res = requests.post(
        f"{api_url}/run_h2_noise_test", json=payload, timeout=600, verify=False
    )
    if res.status_code != 200:
        raise RuntimeError(f"Server returned HTTP {res.status_code}")

    depth_ciphers = res.json()["depth_ciphers"]
    noise_log = []

    # Attempt decryption on each ciphertext to verify noise budget headroom
    for item in depth_ciphers:
        cipher_bytes = base64.b64decode(item["cipher_b64"])
        try:
            _ = fhe_client.try_decrypt_single(cipher_bytes)
            status, err = "Success (Decryption OK)", "N/A"
        except (ValueError, RuntimeError) as e:
            status, err = "Failed (Noise Exhausted)", str(e)

        noise_log.append(
            {
                "Conjunction_Depth": item["clause_conjunction_depth"],
                "Expression": item["expression"],
                "Ciphertext_Size_Bytes": item["cipher_size_bytes"],
                "Decryption_Status": status,
                "Error_Log": err,
            }
        )

    # Persist benchmark result payload to server storage
    if noise_log:
        enc_blob = fhe_client.encrypt_benchmark_data({"h2": noise_log})
        requests.post(
            f"{api_url}/save_encrypted_benchmark",
            json={"benchmark_id": "h2_latest", "encrypted_blob": enc_blob},
            timeout=600,
            verify=False,
        )

    return pd.DataFrame(noise_log)


def execute_h3_scalability_benchmark(
    search_query, target_max, fhe_client, global_vocab, api_url, progress_callback=None
):
    """Executes H3 volume scaling and noise invariance benchmark."""
    pub_bytes = fhe_client.get_public_context_bytes()
    pub_b64 = base64.b64encode(pub_bytes).decode("utf-8")
    sample_term = global_vocab[0] if global_vocab else "test"
    query_str = search_query if search_query else sample_term

    dnf_clauses, _ = fhe_client.encrypt_dnf_vectors(query_str, global_vocab)

    step_size = max(1, target_max // 5)
    sizes = list(range(step_size, target_max + 1, step_size))
    raw_scale = []

    for idx, size in enumerate(sizes):
        if progress_callback:
            progress_callback(idx, len(sizes), size, target_max)

        payload = {
            "public_context_b64": pub_b64,
            "dnf_clauses": dnf_clauses,
            "dataset_sizes": [size],
        }

        res = requests.post(
            f"{api_url}/run_h3_scaling_test", json=payload, timeout=600, verify=False
        )
        if res.status_code == 200:
            chunk_metrics = res.json()["scaling_metrics"]
            raw_scale.extend(chunk_metrics)
        else:
            raise RuntimeError(f"Error for size N={size}: HTTP {res.status_code}")

    # Persist benchmark result payload to server storage
    if raw_scale:
        enc_blob = fhe_client.encrypt_benchmark_data({"h3": raw_scale})
        requests.post(
            f"{api_url}/save_encrypted_benchmark",
            json={"benchmark_id": "h3_latest", "encrypted_blob": enc_blob},
            timeout=600,
            verify=False,
        )

    return pd.DataFrame(raw_scale)


def calculate_h1_statistics(df_h1):
    """Calculates Welch's t-test for Hypothesis 1 latency and RAM metrics."""
    if df_h1 is None or df_h1.empty:
        return None

    t_lat, p_lat = stats.ttest_ind(
        df_h1["Vector_DNF_Latency_ms"], df_h1["Plaintext_Latency_ms"], equal_var=False
    )
    t_ram, p_ram = stats.ttest_ind(
        df_h1["Vector_DNF_RAM_MB"], df_h1["Plaintext_RAM_MB"], equal_var=False
    )

    return {
        "latency": {"t_stat": t_lat, "p_val": p_lat, "reject": p_lat < 0.05},
        "ram": {"t_stat": t_ram, "p_val": p_ram, "reject": p_ram < 0.05},
    }


def calculate_h2_statistics(df_h2):
    """Calculates linear regression & break depth for Hypothesis 2."""
    if df_h2 is None or df_h2.empty:
        return None

    slope, intercept, r_val, p_val, _ = stats.linregress(
        df_h2["Conjunction_Depth"], df_h2["Ciphertext_Size_Bytes"]
    )
    failures = df_h2[df_h2["Decryption_Status"].str.contains("Failed")]
    break_point = failures["Conjunction_Depth"].iloc[0] if not failures.empty else None

    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_val**2,
        "p_val": p_val,
        "break_point": break_point,
        "reject": break_point is not None,
    }


def calculate_h3_statistics(df_h3):
    """Calculates Pearson correlation & OLS noise growth slope for Hypothesis 3."""
    if df_h3 is None or df_h3.empty:
        return None

    r_lat, p_lat = stats.pearsonr(
        df_h3["Dataset_Size_N"], df_h3["Vector_DNF_Latency_ms"]
    )
    slope_noise, _, r_val_noise, p_val_noise, _ = stats.linregress(
        df_h3["Dataset_Size_N"], df_h3["Mean_Ciphertext_Size_Bytes"]
    )

    return {
        "latency": {"r": r_lat, "p_val": p_lat, "reject": p_lat < 0.05},
        "noise": {
            "slope": slope_noise,
            "r_squared": r_val_noise**2,
            "p_val": p_val_noise,
            "reject": not (p_val_noise >= 0.05 or abs(slope_noise) < 1e-3),
        },
    }


# 5. TOP-LEVEL INITIALIZATION BLOCK
if IS_RUNTIME:
    # Initialize client and load datasets using Streamlit resource caching
    initialize_remote_client_cached = st.cache_resource(initialize_remote_client)
    global_vocabulary, client = initialize_remote_client_cached(SERVER_URL)

    load_email_metadata_cached = st.cache_data(load_email_metadata)
    df_full = load_email_metadata_cached(SERVER_URL)

    persisted_benchmarks = load_persisted_benchmarks_from_server(SERVER_URL, client)
else:
    global_vocabulary, client = [], None
    df_full = pd.DataFrame(
        [
            {
                "doc_id": i,
                "file": f"email_{i}.txt",
                "subject": f"Encrypted Email #{i}",
                "body": "[Encrypted Payload]",
            }
            for i in range(50)
        ]
    )
    persisted_benchmarks = {}

# Initialize Streamlit Session State variables
if "search_query" not in st.session_state:
    st.session_state["search_query"] = ""
if "fhe_search_metrics" not in st.session_state:
    st.session_state["fhe_search_metrics"] = None
if "selected_email_idx" not in st.session_state:
    st.session_state["selected_email_idx"] = 0
if "num_rows" not in st.session_state:
    st.session_state["num_rows"] = 100
if "display_limit" not in st.session_state:
    st.session_state["display_limit"] = 20

# Initialize metric cache states from persisted server benchmarks
if "h1_metrics" not in st.session_state:
    st.session_state["h1_metrics"] = persisted_benchmarks.get("h1_metrics", None)
if "h2_metrics" not in st.session_state:
    st.session_state["h2_metrics"] = persisted_benchmarks.get("h2_metrics", None)
if "h3_metrics" not in st.session_state:
    st.session_state["h3_metrics"] = persisted_benchmarks.get("h3_metrics", None)

df_emails = df_full.iloc[: st.session_state["num_rows"]].reset_index(drop=True)


# 6. STREAMLIT INTERACTIVE UI (IS_RUNTIME GUARDED)
if IS_RUNTIME:
    st.sidebar.title("Menu")
    experiment_tab = st.sidebar.radio(
        "Select Tab:",
        [
            "Email Inbox Search",
            "Hypothesis 1: Computational Overhead",
            "Hypothesis 2: Noise Budget Depletion",
            "Hypothesis 3: Dataset Scalability",
            "Inferential Statistical Reports",
        ],
    )

    if experiment_tab == "Email Inbox Search":
        st.title("Email Inbox Search")

        # Callback handlers for slider and input field updates
        def on_slider_change():
            st.session_state["num_rows"] = st.session_state["num_rows_slider"]
            st.session_state["fhe_search_metrics"] = None
            st.session_state["display_limit"] = 20

        def on_query_change():
            st.session_state["search_query"] = st.session_state["search_input_widget"]
            st.session_state["fhe_search_metrics"] = None

        st.slider(
            "Dataset Size:",
            min_value=5,
            max_value=max(5, len(df_full)),
            value=min(st.session_state["num_rows"], len(df_full)),
            step=5,
            key="num_rows_slider",
            on_change=on_slider_change,
            help="Active number of searchable email documents.",
        )

        col_query, col_btn = st.columns([5, 1], vertical_alignment="bottom")
        with col_query:
            if "search_input_widget" not in st.session_state:
                st.session_state["search_input_widget"] = st.session_state[
                    "search_query"
                ]

            user_input = st.text_input(
                "Enter Search Term(s):",
                key="search_input_widget",
                on_change=on_query_change,
                help="You may use boolean operators AND, OR, NOT (case-insensitive) to construct complex queries.",
            )
        with col_btn:
            if st.button("Search", width="stretch"):
                st.session_state["search_query"] = st.session_state[
                    "search_input_widget"
                ]
                st.session_state["fhe_search_metrics"] = None

        # Execute search if metric results are missing
        if st.session_state["fhe_search_metrics"] is None:
            with st.spinner("Searching..."):
                st.session_state["display_limit"] = 20
                st.session_state["selected_email_idx"] = 0
                if "email_radio_key" in st.session_state:
                    del st.session_state["email_radio_key"]

                metrics_res, err_msg = execute_fhe_search(
                    st.session_state["search_query"],
                    st.session_state["num_rows"],
                    client,
                    global_vocabulary,
                    SERVER_URL,
                )
                if err_msg:
                    st.error(err_msg)
                st.session_state["fhe_search_metrics"] = metrics_res

        # Render performance metrics dashboard
        metrics = st.session_state["fhe_search_metrics"]
        if metrics:
            st.divider()
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)

            with m_col1:
                st.caption("Active Query")
                st.code(f"'{metrics['query']}'", language=None)

            with m_col2:
                st.caption("Latency")
                st.code(f"{metrics['eval_time_ms']:.2f} ms", language=None)

            with m_col3:
                st.caption("RAM Consumption")
                st.code(f"{metrics.get('ram_usage_mb', 0.0):.2f} MB", language=None)

            with m_col4:
                st.caption("Ciphertext Size")
                st.code(
                    f"{metrics.get('cipher_size_bytes', 0) / 1024:.2f} KB",
                    language=None,
                )

            if metrics.get("ignored_terms"):
                ignored_str = ", ".join([f"'{w}'" for w in metrics["ignored_terms"]])
                st.warning(
                    f"**Ignored Search Terms:** {ignored_str} — "
                    f"These terms were ignored because they are common stopwords or missing from the search index."
                )

            st.divider()

            # Filter email inbox display to matched search hits
            df_emails["fhe_match"] = metrics["matches"]
            df_matched = df_emails[df_emails["fhe_match"] > 0].reset_index(drop=True)

            col_list, col_preview = st.columns([1, 1.2])
            with col_list:
                st.subheader(f"Emails ({len(df_matched)})")
                if not df_matched.empty:
                    visible_matched = df_matched.iloc[
                        : st.session_state["display_limit"]
                    ]
                    options = [
                        f"[{row['file']}]  {row['subject']}"
                        for _, row in visible_matched.iterrows()
                    ]

                    if st.session_state["selected_email_idx"] >= len(visible_matched):
                        st.session_state["selected_email_idx"] = 0

                    with st.container(height=450):
                        selected_option = st.radio(
                            "Select Email:",
                            options=options,
                            index=st.session_state["selected_email_idx"],
                            key="email_radio_key",
                        )

                    selected_local_idx = (
                        options.index(selected_option)
                        if selected_option in options
                        else 0
                    )
                    st.session_state["selected_email_idx"] = selected_local_idx

                    if len(df_matched) > st.session_state["display_limit"]:
                        col_btn_more, col_info = st.columns([1, 1])
                        with col_btn_more:
                            if st.button("Load More Emails"):
                                st.session_state["display_limit"] += 20
                                st.rerun()
                        with col_info:
                            st.caption(
                                f"Showing {len(visible_matched)} of {len(df_matched)} messages"
                            )
                else:
                    st.info("No matching documents found for active query.")

            with col_preview:
                st.subheader("Email Preview")
                if not df_matched.empty:
                    selected_row = df_matched.iloc[
                        st.session_state["selected_email_idx"]
                    ]
                    doc_id = selected_row["doc_id"]

                    try:
                        # Fetch and decrypt requested email document payload
                        res = requests.get(
                            f"{SERVER_URL}/get_encrypted_email/{doc_id}",
                            timeout=600,
                            verify=False,
                        )
                        if res.status_code == 200:
                            payload_data = res.json()["ciphertext"]
                            aes_ciphertext = (
                                payload_data.get("ciphertext_b64")
                                if isinstance(payload_data, dict)
                                else payload_data
                            )

                            decrypted_raw = client.decrypt_payload(aes_ciphertext)
                            with st.container(border=True, height=450):
                                st.caption(
                                    f"AES-128 Payload Blob (ID {doc_id}): `{aes_ciphertext[:45]}...`"
                                )
                                st.divider()
                                highlighted_body = highlight_text(
                                    decrypted_raw, metrics.get("valid_terms", [])
                                ).replace("\n", "<br>")
                                st.markdown(
                                    f"<div style='max-height: 280px; overflow-y: auto;'>{highlighted_body}</div>",
                                    unsafe_allow_html=True,
                                )
                    except (
                        requests.exceptions.RequestException,
                        json.JSONDecodeError,
                        KeyError,
                        ValueError,
                    ) as e:
                        logger.error(
                            f"Failed to fetch or decrypt email ID {doc_id}: {e}"
                        )
                        st.error(f"Failed to fetch or decrypt payload: {e!s}")

    elif experiment_tab == "Hypothesis 1: Computational Overhead":
        st.title("Hypothesis 1: Computational Overhead")
        st.subheader("Plaintext vs BFV Search Latency & Memory Cost Comparison")

        if st.button("Run H1 Benchmark"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(idx, total, size, target_max):
                status_text.text(
                    f"Benchmarking dataset size N = {size} of {target_max}..."
                )
                progress_bar.progress((idx + 1) / total)

            try:
                df_h1_res = execute_h1_benchmark_step(
                    st.session_state["num_rows"],
                    client,
                    global_vocabulary,
                    SERVER_URL,
                    update_progress,
                )
                st.session_state["h1_metrics"] = df_h1_res
                st.success("H1 Benchmark completed successfully!")
            except (
                requests.exceptions.RequestException,
                RuntimeError,
                ValueError,
            ) as e:
                logger.error(f"H1 Benchmark failed: {e}")
                st.error(f"Benchmark execution failed: {e}")
            finally:
                status_text.empty()
                progress_bar.empty()

        # Render H1 metric data tables and comparative charts
        if st.session_state["h1_metrics"] is not None:
            df_metrics = st.session_state["h1_metrics"]
            col_t, col_c = st.columns([1.2, 1])
            with col_t:
                st.dataframe(df_metrics, width="stretch")

                csv_bytes = convert_df_to_csv(df_metrics)
                st.download_button(
                    label=" Download H1 Benchmark CSV",
                    data=csv_bytes,
                    file_name="h1_fhe_latency_benchmark.csv",
                    mime="text/csv",
                    width="stretch",
                )
            with col_c:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

                ax1.plot(
                    df_metrics["Dataset_Size_N"],
                    df_metrics["Plaintext_Latency_ms"],
                    marker="o",
                    label="Plaintext",
                )
                ax1.plot(
                    df_metrics["Dataset_Size_N"],
                    df_metrics["Vector_DNF_Latency_ms"],
                    marker="s",
                    label="BFV",
                )
                ax1.set_xlabel("Dataset Size (N)")
                ax1.set_ylabel("Latency (ms)")
                ax1.set_title("Execution Latency")
                ax1.legend()

                ax2.plot(
                    df_metrics["Dataset_Size_N"],
                    df_metrics["Plaintext_RAM_MB"],
                    marker="o",
                    label="Plaintext",
                    color="green",
                )
                ax2.plot(
                    df_metrics["Dataset_Size_N"],
                    df_metrics["Vector_DNF_RAM_MB"],
                    marker="s",
                    label="BFV",
                    color="orange",
                )
                ax2.set_xlabel("Dataset Size (N)")
                ax2.set_ylabel("Peak RAM Usage (MB)")
                ax2.set_title("Memory Consumption")
                ax2.legend()

                fig.tight_layout()
                st.pyplot(fig)

    elif experiment_tab == "Hypothesis 2: Noise Budget Depletion":
        st.title("Hypothesis 2: Noise Budget Depletion")
        st.subheader("Ciphertext Size and Decryption Status Across Conjunction Depths")

        if st.button("Run H2 Benchmark"):
            with st.spinner("Evaluating multi-term clause depth noise exhaustion..."):
                try:
                    df_h2_res = execute_h2_noise_test(client, SERVER_URL)
                    st.session_state["h2_metrics"] = df_h2_res
                    st.success(
                        "H2 Noise test completed! Encrypted state persisted to server."
                    )
                except (
                    requests.exceptions.RequestException,
                    RuntimeError,
                    ValueError,
                ) as e:
                    logger.error(f"H2 Noise Test failed: {e}")
                    st.error(f"H2 Noise Test Failed: {e}")

        # Render H2 metric data tables and failure boundary charts
        if st.session_state["h2_metrics"] is not None:
            df_noise = st.session_state["h2_metrics"]
            col_table, col_chart = st.columns([1.2, 1])
            with col_table:
                st.dataframe(df_noise, width="stretch")

                csv_bytes = convert_df_to_csv(df_noise)
                st.download_button(
                    label=" Download H2 Benchmark CSV",
                    data=csv_bytes,
                    file_name="h2_fhe_noise_boundary.csv",
                    mime="text/csv",
                    width="stretch",
                )
            with col_chart:
                fig, ax = plt.subplots(figsize=(5, 3.5))
                colors = [
                    "green" if "Success" in s else "red"
                    for s in df_noise["Decryption_Status"]
                ]
                ax.bar(
                    df_noise["Conjunction_Depth"].astype(str),
                    df_noise["Ciphertext_Size_Bytes"],
                    color=colors,
                )
                ax.set_xlabel("Conjunction Depth")
                ax.set_ylabel("Ciphertext Size (Bytes)")
                ax.set_title("Conjunction Depth Boundary")

                import matplotlib.patches as mpatches

                legend_handles = []
                if any("green" in c for c in colors):
                    legend_handles.append(
                        mpatches.Patch(
                            color="green", label="Decryption Status: Success"
                        )
                    )
                if any("red" in c for c in colors):
                    legend_handles.append(
                        mpatches.Patch(color="red", label="Decryption Status: Failed")
                    )

                if legend_handles:
                    ax.legend(
                        handles=legend_handles, loc="lower left", fontsize="small"
                    )

                st.pyplot(fig)

    elif experiment_tab == "Hypothesis 3: Dataset Scalability":
        st.title("Hypothesis 3: Dataset Scalability")
        st.subheader("Latency Cost & Ciphertext Size Across Document Volumes")

        if st.button("Run H3 Benchmark"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(idx, total, size, target_max):
                status_text.text(
                    f"Scaling evaluation across document volume N = {size} of {target_max}..."
                )
                progress_bar.progress((idx + 1) / total)

            try:
                df_h3_res = execute_h3_scalability_benchmark(
                    st.session_state["search_query"],
                    st.session_state["num_rows"],
                    client,
                    global_vocabulary,
                    SERVER_URL,
                    update_progress,
                )
                st.session_state["h3_metrics"] = df_h3_res
                st.success("H3 Scalability benchmark completed successfully!")
            except (
                requests.exceptions.RequestException,
                RuntimeError,
                ValueError,
            ) as e:
                logger.error(f"H3 Benchmark failed: {e}")
                st.error(f"H3 Benchmark Failed: {e}")
            finally:
                status_text.empty()
                progress_bar.empty()

        # Render H3 scaling data tables and dual-axis growth charts
        if st.session_state["h3_metrics"] is not None:
            df_scale = st.session_state["h3_metrics"]

            formatted_df = df_scale.copy()
            formatted_df = formatted_df.rename(
                columns={
                    "Dataset_Size_N": "Document Volume (N)",
                    "Vector_DNF_Latency_ms": "Latency (ms)",
                    "Mean_Ciphertext_Size_Bytes": "Mean Payload Size (Bytes)",
                }
            )

            col1, col2 = st.columns([1.2, 1])
            with col1:
                st.dataframe(formatted_df, width="stretch")

                csv_bytes = convert_df_to_csv(df_scale)
                st.download_button(
                    label=" Download H3 Benchmark CSV",
                    data=csv_bytes,
                    file_name="h3_fhe_scalability_metrics.csv",
                    mime="text/csv",
                    width="stretch",
                )
            with col2:
                fig, ax1 = plt.subplots(figsize=(5.5, 3.8))

                color_lat = "#1f77b4"
                ax1.set_xlabel("Documents (N)")
                ax1.set_ylabel("Latency (ms)", color=color_lat)
                line1 = ax1.plot(
                    df_scale["Dataset_Size_N"],
                    df_scale["Vector_DNF_Latency_ms"],
                    color=color_lat,
                    marker="o",
                    linewidth=2,
                    label="Latency",
                )
                ax1.tick_params(axis="y", labelcolor=color_lat)

                ax2 = ax1.twinx()
                color_bytes = "#2ca02c"
                ax2.set_ylabel("Mean Ciphertext Size (Bytes)", color=color_bytes)
                line2 = ax2.plot(
                    df_scale["Dataset_Size_N"],
                    df_scale["Mean_Ciphertext_Size_Bytes"],
                    color=color_bytes,
                    linestyle="--",
                    marker="s",
                    linewidth=2,
                    label="Mean Ciphertext Size",
                )
                ax2.tick_params(axis="y", labelcolor=color_bytes)

                mean_val = df_scale["Mean_Ciphertext_Size_Bytes"].mean()
                ax2.set_ylim(mean_val * 0.8, mean_val * 1.2)

                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax1.legend(lines, labels, loc="upper left")

                fig.tight_layout()
                st.pyplot(fig)

    elif experiment_tab == "Inferential Statistical Reports":
        st.title("Inferential Statistical Reports")

        # Report statistical conclusions for Hypothesis 1
        st.header("Hypothesis 1: Computational Overhead")
        h1_stats = calculate_h1_statistics(st.session_state.get("h1_metrics"))
        if h1_stats:
            st.subheader(r"$\text{H}_{1a}$: Execution Latency")
            st.markdown(
                f"* **Welch's t-statistic**: `{h1_stats['latency']['t_stat']:.4f}`"
            )
            st.markdown(f"* **p-value**: `{h1_stats['latency']['p_val']:.6f}`")
            if h1_stats["latency"]["reject"]:
                st.error(
                    r"**Reject $\text{H}_{01a}$**: BVF evaluation incurs statistically significant execution latency overhead compared to plaintext baselines."
                )
            else:
                st.success(
                    r"**Fail to Reject $\text{H}_{01a}$**: No significant latency overhead detected."
                )

            st.subheader(r"$\text{H}_{1b}$: Memory Consumption")
            st.markdown(f"* **Welch's t-statistic**: `{h1_stats['ram']['t_stat']:.4f}`")
            st.markdown(f"* **p-value**: `{h1_stats['ram']['p_val']:.6f}`")
            if h1_stats["ram"]["reject"]:
                st.error(
                    r"**Reject $\text{H}_{01b}$**: BVF evaluation incurs statistically significant memory overhead compared to plaintext baselines."
                )
            else:
                st.success(
                    r"**Fail to Reject $\text{H}_{01b}$**: No significant memory overhead detected."
                )
        else:
            st.warning("Please run the H1 benchmark in Tab 2 first.")

        # Report statistical conclusions for Hypothesis 2
        st.header("Hypothesis 2: Noise Budget Depletion")
        h2_stats = calculate_h2_statistics(st.session_state.get("h2_metrics"))
        if h2_stats:
            st.markdown(
                f"* **Regression Model**: Payload Size = `{h2_stats['slope']:.2f}` × Depth + `{h2_stats['intercept']:.2f}`"
            )
            st.markdown(
                f"* **R²**: `{h2_stats['r_squared']:.4f}` (p = `{h2_stats['p_val']:.6f}`)"
            )

            if h2_stats["reject"]:
                st.error(
                    rf"**Reject $\text{{H}}_{{02}}$**: Conjunction expansion is boundedly constrained. "
                    f"Noise depletion failure occurs at depth **{h2_stats['break_point']}**."
                )
            else:
                st.success(
                    r"**Fail to Reject $\text{H}_{02}$**: No decryption failures detected in tested clause depth range."
                )
        else:
            st.warning("Please run the H2 benchmark in Tab 3 first.")

        # Report statistical conclusions for Hypothesis 3
        st.header("Hypothesis 3: Dataset Scalability")
        h3_stats = calculate_h3_statistics(st.session_state.get("h3_metrics"))
        if h3_stats:
            st.subheader(r"$\text{H}_{3a}$: Latency Cost")
            st.markdown(
                f"* **Pearson Correlation ($r$)**: `{h3_stats['latency']['r']:.6f}`"
            )
            st.markdown(f"* **p-value**: `{h3_stats['latency']['p_val']:.6e}`")
            if h3_stats["latency"]["reject"]:
                st.error(
                    r"**Reject $\text{H}_{03a}$**: Search execution latency displays strong linear scaling $\mathcal{O}(N)$ with document volume."
                )
            else:
                st.success(
                    r"**Fail to Reject $\text{H}_{03a}$**: No significant latency scaling detected."
                )

            st.subheader(r"$\text{H}_{3b}$: Ciphertext Size")
            st.markdown(
                f"* **OLS Slope ($\\beta_1$)**: `{h3_stats['noise']['slope']:.6f}` Bytes/Doc"
            )
            st.markdown(
                f"* **$R^2$ Score**: `{h3_stats['noise']['r_squared']:.4f}` (p = `{h3_stats['noise']['p_val']:.4f}`)"
            )
            if not h3_stats["noise"]["reject"]:
                st.success(
                    r"**Fail to Reject $\text{H}_{03b}$**: Confirms $\mathcal{O}(1)$ noise invariance. Document volume $N$ introduces zero noise expansion."
                )
            else:
                st.error(
                    r"**Reject $\text{H}_{03b}$**: Noise consumption expands with dataset volume."
                )
        else:
            st.warning("Please run the H3 benchmark in Tab 4 first.")
