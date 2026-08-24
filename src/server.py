import base64
import json
import logging
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import tenseal as ts
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.sparse import load_npz

# Directory paths setup for local application data storage
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Global logging configuration
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
ENCRYPTED_DB_FILE = DATA_DIR / "encrypted_emails_db.json"

# Initialize FastAPI application instance
app = FastAPI(title="BFV Searchable Encryption Server")

# Global variable to cache the term-document occurrence matrix in memory
occurrence_matrix = None


@lru_cache(maxsize=16)
def get_cached_context(context_b64: str) -> ts.Context:
    """
    Deserializes and caches a TenSEAL FHE context object from a Base64-encoded string.

    Implements LRU caching to prevent redundant deserialization overhead on repeated queries.
    """
    try:
        context_bytes = base64.b64decode(context_b64)
        return ts.context_from(context_bytes)
    except (TypeError, ValueError, Exception) as e:
        get_cached_context.cache_clear()
        logger.exception("Failed to deserialize TenSEAL Context.")
        raise ValueError(f"TenSEAL Context Deserialization Failed: {e!s}") from e


def load_db_from_disk() -> dict:
    """Loads encrypted email records from the local JSON database file."""
    if os.path.exists(ENCRYPTED_DB_FILE):
        try:
            with open(ENCRYPTED_DB_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.exception(f"Failed to read encrypted DB file ({ENCRYPTED_DB_FILE})")
            return {}
    return {}


# Initialize global memory store for encrypted emails
encrypted_email_store = load_db_from_disk()


def safe_serialize_vector(cipher_vec) -> str:
    """
    Safely serializes a TenSEAL BFV vector ciphertext into a Base64 string.

    Handles transparent ciphertexts by falling back to serializing a zero vector of equal dimension.
    """
    try:
        if hasattr(cipher_vec, "is_transparent") and cipher_vec.is_transparent():
            context = cipher_vec.context()
            zero_vec = ts.bfv_vector(context, [0] * cipher_vec.size())
            return base64.b64encode(zero_vec.serialize()).decode("utf-8")

        return base64.b64encode(cipher_vec.serialize()).decode("utf-8")
    except Exception as e:
        # Check if error is related to transparency or standard serialization
        if "transparent" in str(e).lower():
            logger.warning("Handled transparent ciphertext serialization fallback.")
            context = cipher_vec.context()
            zero_vec = ts.bfv_vector(context, [0] * cipher_vec.size())
            return base64.b64encode(zero_vec.serialize()).decode("utf-8")

        logger.exception("Serialization failed for ciphertext vector.")
        raise RuntimeError(f"Ciphertext serialization error: {e}") from e


# Threshold for triggering system RAM guard (85.0%)
MAX_MEMORY_PERCENT_THRESHOLD = 85.0


def verify_system_memory_safety():
    """
    Checks current server memory usage against the maximum threshold.

    Raises HTTP 507 Insufficient Storage if system RAM usage exceeds limit.
    """
    mem = psutil.virtual_memory()
    if mem.percent >= MAX_MEMORY_PERCENT_THRESHOLD:
        logger.warning(
            f"Memory threshold exceeded: {mem.percent}% used (Limit: {MAX_MEMORY_PERCENT_THRESHOLD}%)"
        )
        raise HTTPException(
            status_code=507,
            detail=(
                f"Operation aborted by server memory guard. Current system RAM usage "
                f"is at {mem.percent:.1f}%, exceeding the safe threshold of {MAX_MEMORY_PERCENT_THRESHOLD}%."
            ),
        )


# Path to stored benchmark data payloads
ENCRYPTED_BENCHMARK_FILE = DATA_DIR / "encrypted_benchmarks_db.json"


def load_benchmark_db_from_disk() -> dict:
    """Loads persistent encrypted benchmark records from disk."""
    if os.path.exists(ENCRYPTED_BENCHMARK_FILE):
        try:
            with open(ENCRYPTED_BENCHMARK_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.exception("Failed to load benchmark DB from disk.")
            return {}
    return {}


# Initialize global memory store for benchmark data
ENCRYPTED_BENCHMARK_STORE = load_benchmark_db_from_disk()


# Pydantic Data Models


class EncryptedStoragePayload(BaseModel):
    """Schema for storing arbitrary opaque benchmark data."""

    benchmark_id: str
    encrypted_blob: str


class UniversalSearchPayload(BaseModel):
    """Schema for processing homomorphic search operations."""

    public_context_b64: str
    target_doc_count: int | None = None
    dnf_clauses: list[dict[str, Any]]


class H1BenchmarkPayload(BaseModel):
    """Schema for running latency and memory comparison benchmarks across dataset sizes."""

    public_context_b64: str
    dnf_clauses: list[dict[str, Any]]
    dataset_sizes: list[int]


class H2NoisePayload(BaseModel):
    """Schema for analyzing noise growth across varying conjunction term lengths."""

    public_context_b64: str
    num_clause_terms: int = 5


class H3ScalingPayload(BaseModel):
    """Schema for evaluating performance scaling across larger dataset subsets."""

    public_context_b64: str
    dnf_clauses: list[dict[str, Any]]
    dataset_sizes: list[int] | None = None


# Core FHE Processing Functions


def evaluate_boolean_query_vector(
    enc_query: ts.BFVVector, matrix
) -> list[ts.BFVVector]:
    """
    Computes homomorphic dot products between an encrypted query term vector and document vectors.

    Returns a list of encrypted score scalars corresponding to each document in the matrix segment.
    """
    doc_scores = []
    _, num_docs = matrix.shape

    dense_matrix = matrix.toarray() if hasattr(matrix, "toarray") else np.array(matrix)
    q_size = enc_query.size()

    for doc_idx in range(num_docs):
        doc_column = dense_matrix[:, doc_idx].astype(int).tolist()

        # Align document term vector length to match the query vector dimension
        if len(doc_column) > q_size:
            doc_column = doc_column[:q_size]
        elif len(doc_column) < q_size:
            doc_column = doc_column + [0] * (q_size - len(doc_column))

        # Perform homomorphic element-wise multiplication and sum accumulation
        score_vector = enc_query * doc_column
        score_cipher = score_vector.sum()
        doc_scores.append(score_cipher)

    return doc_scores


def evaluate_dnf_vectors(
    dnf_clauses: list[dict[str, Any]], context: ts.Context, matrix
) -> list[dict[str, Any]]:
    """
    Evaluates DNF query clauses over the term-document matrix homomorphically.

    Processes positive terms homomorphically and checks negative term exclusions in plaintext on the server.
    """
    dnf_eval_results = []
    num_docs = matrix.shape[1]

    for clause in dnf_clauses:
        try:
            enc_pos_bytes = base64.b64decode(clause["enc_pos_b64"])
            enc_pos_vec = ts.bfv_vector_from(context, enc_pos_bytes)
        except KeyError as e:
            raise ValueError(f"Missing key in clause specification: {e}") from e
        except Exception as e:
            raise ValueError(f"Failed to decode BFV vector clause: {e}") from e

        is_pure_neg = clause.get("is_pure_negation", False)
        clause_scores = evaluate_boolean_query_vector(enc_pos_vec, matrix)
        neg_indices = clause.get("neg_indices", [])

        final_clause_scores = []
        for doc_i in range(num_docs):
            has_negated_term = False

            # Plaintext evaluation of prohibited negative terms
            if neg_indices:
                for neg_idx in neg_indices:
                    if 0 <= neg_idx < matrix.shape[0]:
                        val = matrix[neg_idx, doc_i]
                        if hasattr(val, "toarray"):
                            val = val.toarray()[0][0]
                        if val == 1:
                            has_negated_term = True
                            break

            # Zero out score if document contains a negated term
            if has_negated_term:
                final_clause_scores.append(ts.bfv_vector(context, [0]))
            elif is_pure_neg:
                final_clause_scores.append(ts.bfv_vector(context, [1]))
            else:
                final_clause_scores.append(clause_scores[doc_i])

        serialized_scores = [safe_serialize_vector(c) for c in final_clause_scores]
        dnf_eval_results.append(
            {
                "required_pos_count": clause.get("required_pos_count", 0),
                "is_pure_negation": is_pure_neg,
                "encrypted_scores": serialized_scores,
            }
        )

    return dnf_eval_results


def get_occurrence_matrix():
    """Lazy loader for the sparse term-document occurrence matrix from disk."""
    global occurrence_matrix
    if occurrence_matrix is None:
        matrix_path = DATA_DIR / "occurrence_matrix_sparse.npz"
        if os.path.exists(matrix_path):
            try:
                occurrence_matrix = load_npz(matrix_path)
            except Exception as e:
                logger.exception(f"Corrupted or invalid matrix file at {matrix_path}.")
                raise RuntimeError(f"Unable to load sparse matrix file: {e}") from e
        else:
            raise FileNotFoundError(
                f"Matrix file missing at target location: {matrix_path}"
            )
    return occurrence_matrix


# FastAPI Endpoints


@app.get("/")
def read_root():
    """Healthcheck endpoint to verify server status."""
    return {"status": "ok", "message": "FHE Search Server Running"}


@app.post("/fhe_boolean_search")
def fhe_boolean_search(payload: UniversalSearchPayload):
    """Executes an encrypted vector-based boolean search using DNF clauses."""
    verify_system_memory_safety()
    try:
        t0 = time.perf_counter()
        full_matrix = get_occurrence_matrix()

        # Trim matrix to the requested target document count if specified
        matrix = (
            full_matrix[:, : payload.target_doc_count]
            if payload.target_doc_count
            and payload.target_doc_count <= full_matrix.shape[1]
            else full_matrix
        )

        context = get_cached_context(payload.public_context_b64)
        dnf_results = evaluate_dnf_vectors(payload.dnf_clauses, context, matrix)
        eval_time_ms = (time.perf_counter() - t0) * 1000

        return {
            "status": "success",
            "mode_used": "VECTOR_DNF",
            "circuit_depth_type": "O(1) - Parallel Clause Dot Products",
            "server_processing_time_ms": round(eval_time_ms, 2),
            "dnf_results": dnf_results,
        }
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        logger.warning(f"Bad input or environment issue during search: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Unhandled error during FHE search:")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")


@app.post("/run_h1_benchmark")
def run_h1_benchmark(payload: H1BenchmarkPayload):
    """Runs latency and RAM benchmarking comparing plaintext vs encrypted query evaluations."""
    verify_system_memory_safety()
    try:
        matrix = get_occurrence_matrix()
        context = get_cached_context(payload.public_context_b64)
        process = psutil.Process()

        experimental_results = []
        for size in payload.dataset_sizes:
            verify_system_memory_safety()
            sub_matrix = matrix[:, :size]

            # Measure unencrypted baseline latency and RAM consumption
            mem_plain_before = process.memory_info().rss / (1024 * 1024)
            start_p = time.perf_counter_ns()

            _ = (
                sub_matrix[0, :].toarray()
                if hasattr(sub_matrix, "toarray")
                else sub_matrix[0, :]
            )
            plain_lat = (time.perf_counter_ns() - start_p) / 1e6
            plain_ram = max(
                0.01, (process.memory_info().rss / (1024 * 1024)) - mem_plain_before
            )

            # Measure homomorphic evaluation latency and RAM consumption
            mem_enc_before = process.memory_info().rss / (1024 * 1024)
            start_dnf = time.perf_counter_ns()

            _ = evaluate_dnf_vectors(payload.dnf_clauses, context, sub_matrix)

            dnf_lat = (time.perf_counter_ns() - start_dnf) / 1e6
            dnf_ram = max(
                0.01, (process.memory_info().rss / (1024 * 1024)) - mem_enc_before
            )

            experimental_results.append(
                {
                    "Dataset_Size_N": size,
                    "Plaintext_Latency_ms": plain_lat,
                    "Vector_DNF_Latency_ms": dnf_lat,
                    "Plaintext_RAM_MB": round(plain_ram, 2),
                    "Vector_DNF_RAM_MB": round(dnf_ram, 2),
                }
            )

        return {"metrics": experimental_results}
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("H1 Benchmark Error.")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run_h2_noise_test")
def run_h2_noise_test(payload: H2NoisePayload):
    """Evaluates output ciphertext sizes across increasing conjunction clause lengths."""
    verify_system_memory_safety()
    try:
        matrix = get_occurrence_matrix()
        context = get_cached_context(payload.public_context_b64)
        num_docs = min(50, matrix.shape[1])
        depth_results = []

        vocab_len = min(payload.num_clause_terms, matrix.shape[0])

        for clause_length in range(1, vocab_len + 1):
            query_vec = [1 if i < clause_length else 0 for i in range(matrix.shape[0])]
            enc_q = ts.bfv_vector(context, query_vec)

            sub_matrix = matrix[:, :num_docs]
            scores = evaluate_boolean_query_vector(enc_q, sub_matrix)

            sample_b64 = safe_serialize_vector(scores[0])
            depth_results.append(
                {
                    "clause_conjunction_depth": clause_length,
                    "expression": f"Conjunction of {clause_length} Terms",
                    "cipher_size_bytes": len(base64.b64decode(sample_b64)),
                    "cipher_b64": sample_b64,
                }
            )

        return {"status": "success", "depth_ciphers": depth_results}
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("H2 Noise Test Error.")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run_h3_scaling_test")
def run_h3_scaling_test(payload: H3ScalingPayload):
    """Measures latency scaling and response ciphertext sizes across linearly increasing dataset sizes."""
    verify_system_memory_safety()
    try:
        matrix = get_occurrence_matrix()
        context = get_cached_context(payload.public_context_b64)

        # Default to step-wise dataset size increments if not specified
        if payload.dataset_sizes:
            dataset_sizes = payload.dataset_sizes
        else:
            total_docs = matrix.shape[1]
            step = max(10, total_docs // 5)
            dataset_sizes = list(range(step, total_docs + 1, step))

        scaling_metrics = []
        for size in dataset_sizes:
            sub_matrix = matrix[:, :size]

            start_dnf = time.perf_counter_ns()
            dnf_results = evaluate_dnf_vectors(payload.dnf_clauses, context, sub_matrix)
            dnf_lat = (time.perf_counter_ns() - start_dnf) / 1e6

            all_payload_bytes = [
                len(base64.b64decode(score_b64))
                for score_b64 in dnf_results[0]["encrypted_scores"]
            ]
            mean_payload_bytes = (
                sum(all_payload_bytes) / len(all_payload_bytes)
                if all_payload_bytes
                else 0
            )

            scaling_metrics.append(
                {
                    "Dataset_Size_N": size,
                    "Vector_DNF_Latency_ms": round(dnf_lat, 2),
                    "Mean_Ciphertext_Size_Bytes": round(mean_payload_bytes, 2),
                }
            )

        return {"scaling_metrics": scaling_metrics}
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("H3 Scaling Test Error.")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get_email_metadata")
def get_email_metadata():
    """Retrieves document index metadata and encrypted ciphertext samples for all stored emails."""
    global encrypted_email_store
    if not encrypted_email_store:
        encrypted_email_store = load_db_from_disk()

    metadata = []
    for doc_id, payload in encrypted_email_store.items():
        metadata.append(
            {
                "doc_id": int(doc_id),
                "file": f"email_{doc_id}.txt",
                "subject": f"Encrypted Email #{doc_id}",
                "ciphertext_sample": payload.get("ciphertext_b64", "")
                if isinstance(payload, dict)
                else payload,
            }
        )
    return {"metadata": metadata}


@app.get("/get_vocabulary")
def get_vocabulary():
    """Retrieves the system vocabulary JSON mapping file."""
    vocab_path = DATA_DIR / "vocabulary.json"
    if os.path.exists(vocab_path):
        try:
            with open(vocab_path, "r") as f:
                vocab = json.load(f)
            return {"vocabulary": vocab}
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Corrupted vocabulary file at {vocab_path}: {e}")
            raise HTTPException(
                status_code=500, detail="Vocabulary file corrupted."
            ) from e

    raise HTTPException(status_code=404, detail="Vocabulary file missing on server.")


@app.post("/save_encrypted_benchmark")
def save_encrypted_benchmark(payload: EncryptedStoragePayload):
    """Persists a client-encrypted benchmark payload dictionary to disk."""
    ENCRYPTED_BENCHMARK_STORE[payload.benchmark_id] = payload.encrypted_blob
    try:
        with open(ENCRYPTED_BENCHMARK_FILE, "w") as f:
            json.dump(ENCRYPTED_BENCHMARK_STORE, f, indent=2)
    except (OSError, TypeError) as e:
        logger.error(f"Failed to persist benchmark payload to disk: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to persist benchmark to disk."
        ) from e

    return {
        "status": "success",
        "message": "Opaque ciphertext stored successfully on disk.",
    }


@app.get("/get_encrypted_benchmark/{benchmark_id}")
def get_encrypted_benchmark(benchmark_id: str):
    """Fetches a stored encrypted benchmark record by ID."""
    global ENCRYPTED_BENCHMARK_STORE
    if not ENCRYPTED_BENCHMARK_STORE:
        ENCRYPTED_BENCHMARK_STORE = load_benchmark_db_from_disk()

    if benchmark_id in ENCRYPTED_BENCHMARK_STORE:
        return {"encrypted_blob": ENCRYPTED_BENCHMARK_STORE[benchmark_id]}
    raise HTTPException(status_code=404, detail="Benchmark record not found.")


@app.get("/get_encrypted_email/{doc_id}")
def get_encrypted_email(doc_id: int):
    """Fetches full encrypted ciphertext for a single email document by numeric ID."""
    global encrypted_email_store
    key = str(doc_id)
    if key not in encrypted_email_store:
        encrypted_email_store = load_db_from_disk()
    if key in encrypted_email_store:
        return {"doc_id": doc_id, "ciphertext": encrypted_email_store[key]}
    raise HTTPException(status_code=404, detail=f"Email ID {doc_id} not found.")
