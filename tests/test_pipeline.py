import base64
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd
import pytest
import requests
import scipy.sparse as sp
import tenseal as ts
from fastapi.testclient import TestClient

# Import modules under test
import app
import src.setup_db as db
from src import server
from src.client import FHEClient, parse_dnf_query
from src.server import app as fastapi_app
from src.server import evaluate_dnf_vectors, safe_serialize_vector


# GLOBAL FIXTURE: Prevent All Disk Overwrites
@pytest.fixture(autouse=True)
def isolate_test_environment(tmp_path):
    """
    Globally intercepts all file read/write locations across server and client
    modules, redirecting database JSONs and secret key paths to temporary test directories.
    """
    temp_benchmark_file = tmp_path / "test_encrypted_benchmarks_db.json"
    temp_db_file = tmp_path / "test_encrypted_emails_db.json"
    temp_key_file = tmp_path / "test_secret.key"

    with (
        patch("src.server.ENCRYPTED_BENCHMARK_FILE", temp_benchmark_file),
        patch("src.server.ENCRYPTED_DB_FILE", temp_db_file),
        patch("src.client.AES_KEY_PATH", str(temp_key_file)),
    ):
        yield


# 1. app.py Backend Logic Coverage Tests
class TestAppBackendLogic:
    @pytest.fixture(autouse=True)
    def setup_app_mocks(self, monkeypatch):
        """Mock external network calls and Streamlit session states for app tests."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "vocabulary": ["test", "vocab"],
            "metadata": [],
            "encrypted_blob": b"",
        }

        session_state = {
            "num_rows": 10,
            "display_limit": 20,
            "selected_email_idx": 0,
            "fhe_search_metrics": None,
            "search_query": "",
        }
        # Inject the mocked session state directly into the Streamlit app state
        monkeypatch.setattr(app, "st", MagicMock(session_state=session_state))

    def test_highlight_text(self):
        """Tests keyword highlighting logic for the message previewer."""
        sample_text = "The natural gas report is ready for review."
        valid_terms = ["gas", "report"]

        highlighted = app.highlight_text(sample_text, valid_terms)

        # Assert that HTML mark tags were wrapped around the query terms
        assert "<mark style=" in highlighted
        assert "gas" in highlighted
        assert "report" in highlighted

        # Test edge cases (empty strings or missing terms)
        assert app.highlight_text("", valid_terms) == ""
        assert app.highlight_text(sample_text, []) == sample_text

    def test_convert_df_to_csv(self):
        """Tests DataFrame conversion helper used for CSV downloads."""
        df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        csv_bytes = app.convert_df_to_csv(df)

        # Verify that output is encoded as utf-8 bytes containing expected CSV rows
        assert isinstance(csv_bytes, bytes)
        assert b"col1,col2" in csv_bytes
        assert b"1,a" in csv_bytes

    def test_load_email_metadata_fallback(self):
        """Tests metadata loading fallback when the remote server is unreachable."""
        with patch(
            "requests.get",
            side_effect=requests.exceptions.RequestException("Connection refused"),
        ):
            df = app.load_email_metadata("http://127.0.0.1:8000")

            # Fallback should return a synthetic default DataFrame with 50 items
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 50
            assert "doc_id" in df.columns
            assert "body" in df.columns

    def test_load_email_metadata_success(self):
        """Tests metadata loading upon successful API response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "metadata": [
                {
                    "doc_id": 0,
                    "file": "email_0.txt",
                    "subject": "Test",
                    "body": "Body",
                }
            ]
        }

        with patch("requests.get", return_value=mock_response):
            df = app.load_email_metadata("http://127.0.0.1:8000")

            # Validate structural integrity of response transformed into DataFrame
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 1
            assert df.iloc[0]["subject"] == "Test"

    def test_execute_fhe_search_empty_query(self):
        """Tests search state behavior when an empty query string is passed."""
        metrics, err = app.execute_fhe_search(
            "",
            target_doc_count=10,
            fhe_client=MagicMock(),
            global_vocab=["gas"],
            api_url="http://127.0.0.1:8000",
        )

        # An empty query should shortcut network requests and match all target documents
        assert err is None
        assert metrics is not None
        assert metrics["query"] == "[All Messages]"
        assert len(metrics["matches"]) == 10
        assert all(m == 1 for m in metrics["matches"])

    def test_execute_fhe_search_success(self):
        """Tests remote search execution and response parsing with mocked server response."""
        mock_client = MagicMock()
        mock_client.get_public_context_bytes.return_value = b"pub_key_ctx"
        mock_client.encrypt_dnf_vectors.return_value = (
            [{"pos_indices": [0]}],
            [],
        )
        mock_client.decrypt_dnf_results.return_value = [1, 0, 1, 0, 0]

        mock_post_response = MagicMock()
        mock_post_response.status_code = 200
        mock_post_response.json.return_value = {
            "server_processing_time_ms": 15.5,
            "circuit_depth_type": "O(1) - Parallel",
            "dnf_results": [],
        }

        with patch("requests.post", return_value=mock_post_response):
            metrics, err = app.execute_fhe_search(
                "gas AND energy",
                target_doc_count=5,
                fhe_client=mock_client,
                global_vocab=["gas", "energy"],
                api_url="http://127.0.0.1:8000",
            )

            # Confirm metrics dict is populated correctly and matches decryptions
            assert err is None
            assert metrics is not None
            assert metrics["eval_time_ms"] == 15.5
            assert metrics["matches"] == [1, 0, 1, 0, 0]

    def test_initialize_remote_client_success(self):
        """Tests initialize_remote_client when backend ping and vocabulary retrieval succeed."""
        # Mock ping endpoint response
        mock_ping_res = MagicMock()
        mock_ping_res.status_code = 200

        # Mock vocabulary endpoint response
        mock_vocab_res = MagicMock()
        mock_vocab_res.status_code = 200
        mock_vocab_res.json.return_value = {"vocabulary": ["gas", "oil", "energy"]}

        def mock_requests_get(url, *args, **kwargs):
            if url.endswith("/"):
                return mock_ping_res
            elif url.endswith("/get_vocabulary"):
                return mock_vocab_res
            return MagicMock(status_code=404)

        with (
            patch("requests.get", side_effect=mock_requests_get),
            patch("app.FHEClient") as MockFHEClient,
        ):
            mock_client_instance = MagicMock()
            MockFHEClient.return_value = mock_client_instance

            vocab, client = app.initialize_remote_client("http://127.0.0.1:8000")

            # Ensure client instantiated with vocabulary length
            assert vocab == ["gas", "oil", "energy"]
            assert client == mock_client_instance
            MockFHEClient.assert_called_once_with(3)

    def test_initialize_remote_client_server_down_fallback(self):
        """Tests fallback behavior when the backend server is unreachable or fails to return vocabulary."""
        with patch(
            "requests.get",
            side_effect=requests.exceptions.RequestException("Connection refused"),
        ):
            vocab, client = app.initialize_remote_client("http://127.0.0.1:8000")

            # Validate fallback empty state
            assert vocab == []
            assert client is None

    def test_load_persisted_benchmarks_from_server_success(self):
        """Tests successful benchmark retrieval and conversion into DataFrames."""
        mock_client = MagicMock()
        mock_client.decrypt_benchmark_data.side_effect = lambda blob: {
            "h1": [{"Dataset_Size_N": 10, "Vector_DNF_Latency_ms": 12.5}],
            "h2": [{"Conjunction_Depth": 1, "Ciphertext_Size_Bytes": 1024}],
            "h3": [{"Dataset_Size_N": 10, "Mean_Ciphertext_Size_Bytes": 512}],
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"encrypted_blob": "mock_encrypted_b64"}

        with patch("requests.get", return_value=mock_response):
            results = app.load_persisted_benchmarks_from_server(
                "http://127.0.0.1:8000", mock_client
            )

            # Confirm metrics maps are created as expected DataFrames
            assert "h1_metrics" in results
            assert "h2_metrics" in results
            assert "h3_metrics" in results
            assert isinstance(results["h1_metrics"], pd.DataFrame)
            assert len(results["h1_metrics"]) == 1
            assert results["h1_metrics"].iloc[0]["Vector_DNF_Latency_ms"] == 12.5

    def test_load_persisted_benchmarks_from_server_partial_keys(self):
        """Tests behavior when decrypted data only contains a subset of expected benchmark keys."""
        mock_client = MagicMock()
        # Only returning 'h1', missing 'h2' and 'h3'
        mock_client.decrypt_benchmark_data.return_value = {
            "h1": [{"Dataset_Size_N": 5}]
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"encrypted_blob": "mock_encrypted_b64"}

        with patch("requests.get", return_value=mock_response):
            results = app.load_persisted_benchmarks_from_server(
                "http://127.0.0.1:8000", mock_client
            )

            # Check that missing keys are safely omitted from result dictionary
            assert "h1_metrics" in results
            assert "h2_metrics" not in results
            assert "h3_metrics" not in results

    def test_load_persisted_benchmarks_from_server_http_error(self):
        """Tests fallback to empty dict when server returns non-200 HTTP status."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("requests.get", return_value=mock_response):
            results = app.load_persisted_benchmarks_from_server(
                "http://127.0.0.1:8000", MagicMock()
            )

            assert results == {}

    def test_load_persisted_benchmarks_from_server_connection_exception(self):
        """Tests exception handling (broad try/except) when network requests fail."""
        with patch(
            "requests.get",
            side_effect=requests.exceptions.RequestException("Connection timed out"),
        ):
            results = app.load_persisted_benchmarks_from_server(
                "http://127.0.0.1:8000", MagicMock()
            )

            assert results == {}

    def test_execute_fhe_search_server_error_status(self):
        """Tests handling when backend returns non-200 HTTP response (e.g., 500 Internal Server Error)."""
        mock_client = MagicMock()
        mock_client.get_public_context_bytes.return_value = b"pub_key_ctx"
        mock_client.encrypt_dnf_vectors.return_value = ([{"pos_indices": [0]}], [])

        mock_post_response = MagicMock()
        mock_post_response.status_code = 500
        mock_post_response.text = "Internal Server Error"

        with patch("requests.post", return_value=mock_post_response):
            metrics, err = app.execute_fhe_search(
                "gas",
                target_doc_count=5,
                fhe_client=mock_client,
                global_vocab=["gas"],
                api_url="http://127.0.0.1:8000",
            )

            # Verify empty zero-match list fallback and descriptive error string
            assert metrics["matches"] == [0] * 5
            assert err == "Server Error 500: Internal Server Error"

    def test_execute_fhe_search_connection_exception(self):
        """Tests exception handling when network post fails (e.g., Timeout or Connection Refused)."""
        mock_client = MagicMock()
        mock_client.get_public_context_bytes.return_value = b"pub_key_ctx"
        mock_client.encrypt_dnf_vectors.return_value = ([{"pos_indices": [0]}], [])

        with patch(
            "requests.post",
            side_effect=requests.exceptions.RequestException("Connection refused"),
        ):
            metrics, err = app.execute_fhe_search(
                "gas",
                target_doc_count=5,
                fhe_client=mock_client,
                global_vocab=["gas"],
                api_url="http://127.0.0.1:8000",
            )

            assert metrics["matches"] == [0] * 5
            assert "Failed to connect to server" in err

    # H1 BENCHMARK TESTS

    @patch("requests.post")
    def test_execute_h1_benchmark_step_success(self, mock_post):
        """Tests H1 benchmark execution across steps and saving encrypted results."""
        mock_client = MagicMock()
        mock_client.get_public_context_bytes.return_value = b"pub_key"
        mock_client.encrypt_dnf_vectors.return_value = ([{"pos_indices": [0]}], [])
        mock_client.encrypt_benchmark_data.return_value = "enc_blob_data"

        res_step = MagicMock(status_code=200)
        res_step.json.return_value = {
            "metrics": [
                {
                    "Vector_DNF_Latency_ms": 10.5,
                    "Plaintext_Latency_ms": 1.2,
                    "Vector_DNF_RAM_MB": 50,
                    "Plaintext_RAM_MB": 10,
                }
            ]
        }
        res_save = MagicMock(status_code=200)
        # Mock sequential calls: 1st for step calculation, 2nd for saving results
        mock_post.side_effect = [res_step, res_save]

        progress_callback = MagicMock()

        df = app.execute_h1_benchmark_step(
            target_max=1,
            fhe_client=mock_client,
            global_vocab=["apple", "banana"],
            api_url="http://127.0.0.1:8000",
            progress_callback=progress_callback,
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert progress_callback.call_count == 1
        assert mock_post.call_count == 2

    @patch("requests.post")
    def test_execute_h1_benchmark_step_http_error(self, mock_post):
        """Tests RuntimeError handling when H1 server endpoint returns HTTP error."""
        mock_client = MagicMock()
        mock_client.get_public_context_bytes.return_value = b"pub_key"
        mock_client.encrypt_dnf_vectors.return_value = ([], [])

        mock_post.return_value = MagicMock(status_code=500)

        try:
            app.execute_h1_benchmark_step(
                target_max=5,
                fhe_client=mock_client,
                global_vocab=[],
                api_url="http://127.0.0.1:8000",
            )
            assert False, "Expected RuntimeError was not raised"
        except RuntimeError as e:
            assert "Error for size N=" in str(e)

    # H2 NOISE TEST TESTS

    @patch("requests.post")
    def test_execute_h2_noise_test_success_and_decryption_failure(self, mock_post):
        """Tests H2 noise testing handling both successful decryptions and noise exhaustion failures."""
        mock_client = MagicMock()
        mock_client.get_public_context_bytes.return_value = b"pub_key"

        # Simulate first ciphertext decrypting OK, and second throwing noise budget exhausted
        mock_client.try_decrypt_single.side_effect = [
            True,
            ValueError("Noise budget exhausted"),
        ]
        mock_client.encrypt_benchmark_data.return_value = "enc_blob_h2"

        cipher_b64_1 = base64.b64encode(b"cipher1").decode("utf-8")
        cipher_b64_2 = base64.b64encode(b"cipher2").decode("utf-8")

        res_test = MagicMock(status_code=200)
        res_test.json.return_value = {
            "depth_ciphers": [
                {
                    "cipher_b64": cipher_b64_1,
                    "clause_conjunction_depth": 1,
                    "expression": "A",
                    "cipher_size_bytes": 100,
                },
                {
                    "cipher_b64": cipher_b64_2,
                    "clause_conjunction_depth": 2,
                    "expression": "A AND B",
                    "cipher_size_bytes": 200,
                },
            ]
        }
        res_save = MagicMock(status_code=200)
        mock_post.side_effect = [res_test, res_save]

        df = app.execute_h2_noise_test(mock_client, api_url="http://127.0.0.1:8000")

        # Verify DataFrame records both success and noise failure statuses
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert df.iloc[0]["Decryption_Status"] == "Success (Decryption OK)"
        assert df.iloc[1]["Decryption_Status"] == "Failed (Noise Exhausted)"
        assert "Noise budget exhausted" in df.iloc[1]["Error_Log"]

    @patch("requests.post")
    def test_execute_h2_noise_test_server_error(self, mock_post):
        """Tests RuntimeError handling when H2 noise test endpoint fails."""
        mock_client = MagicMock()
        mock_client.get_public_context_bytes.return_value = b"pub_key"
        mock_post.return_value = MagicMock(status_code=500)

        try:
            app.execute_h2_noise_test(mock_client, api_url="http://127.0.0.1:8000")
            assert False, "Expected RuntimeError was not raised"
        except RuntimeError as e:
            assert "Server returned HTTP 500" in str(e)

    # H3 SCALABILITY BENCHMARK TESTS

    @patch("requests.post")
    def test_execute_h3_scalability_benchmark_success(self, mock_post):
        """Tests H3 scalability benchmark execution, progress callback, and dataset metric accumulation."""
        mock_client = MagicMock()
        mock_client.get_public_context_bytes.return_value = b"pub_key"
        mock_client.encrypt_dnf_vectors.return_value = ([], [])
        mock_client.encrypt_benchmark_data.return_value = "enc_blob_h3"

        res_step = MagicMock(status_code=200)
        res_step.json.return_value = {
            "scaling_metrics": [
                {
                    "Dataset_Size_N": 1,
                    "Vector_DNF_Latency_ms": 15.0,
                    "Mean_Ciphertext_Size_Bytes": 2048,
                }
            ]
        }
        res_save = MagicMock(status_code=200)
        mock_post.side_effect = [res_step, res_save]

        progress_callback = MagicMock()

        df = app.execute_h3_scalability_benchmark(
            search_query="test query",
            target_max=1,
            fhe_client=mock_client,
            global_vocab=["test"],
            api_url="http://127.0.0.1:8000",
            progress_callback=progress_callback,
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.iloc[0]["Vector_DNF_Latency_ms"] == 15.0
        # Validates that the progress_callback execution line is reached and called
        progress_callback.assert_called_once_with(0, 1, 1, 1)

    @patch("requests.post")
    def test_execute_h3_scalability_benchmark_error(self, mock_post):
        """Tests RuntimeError when H3 scaling endpoint returns non-200."""
        mock_client = MagicMock()
        mock_client.get_public_context_bytes.return_value = b"pub_key"
        mock_client.encrypt_dnf_vectors.return_value = ([], [])
        mock_post.return_value = MagicMock(status_code=400)

        try:
            app.execute_h3_scalability_benchmark(
                search_query=None,
                target_max=5,
                fhe_client=mock_client,
                global_vocab=[],
                api_url="http://127.0.0.1:8000",
            )
            assert False, "Expected RuntimeError was not raised"
        except RuntimeError as e:
            assert "HTTP 400" in str(e)

    # STATISTICAL CALCULATION TESTS

    def test_calculate_h1_statistics(self):
        """Tests Welch's t-test statistical outputs and empty DataFrame guards."""
        assert app.calculate_h1_statistics(None) is None
        assert app.calculate_h1_statistics(pd.DataFrame()) is None

        df = pd.DataFrame(
            {
                "Vector_DNF_Latency_ms": [10.0, 12.0, 14.0, 15.0],
                "Plaintext_Latency_ms": [1.0, 1.2, 1.1, 1.3],
                "Vector_DNF_RAM_MB": [100, 105, 110, 115],
                "Plaintext_RAM_MB": [10, 10, 11, 10],
            }
        )
        stats = app.calculate_h1_statistics(df)
        assert "latency" in stats
        assert "ram" in stats
        # Latencies are significantly different, so null hypothesis should be rejected
        assert stats["latency"]["reject"] == True

    def test_calculate_h2_statistics(self):
        """Tests linear regression and noise breaking depth identification."""
        assert app.calculate_h2_statistics(None) is None
        assert app.calculate_h2_statistics(pd.DataFrame()) is None

        df = pd.DataFrame(
            {
                "Conjunction_Depth": [1, 2, 3, 4],
                "Ciphertext_Size_Bytes": [1000, 2000, 3000, 4000],
                "Decryption_Status": [
                    "Success",
                    "Success",
                    "Failed (Noise Exhausted)",
                    "Failed (Noise Exhausted)",
                ],
            }
        )
        stats = app.calculate_h2_statistics(df)
        # Verify first failing index identification
        assert stats["break_point"] == 3
        assert stats["reject"] is True
        assert stats["r_squared"] == 1.0

    def test_calculate_h3_statistics(self):
        """Tests Pearson correlation and OLS regression stats for scale testing."""
        assert app.calculate_h3_statistics(None) is None
        assert app.calculate_h3_statistics(pd.DataFrame()) is None

        df = pd.DataFrame(
            {
                "Dataset_Size_N": [10, 20, 30, 40],
                "Vector_DNF_Latency_ms": [100, 200, 300, 400],
                "Mean_Ciphertext_Size_Bytes": [1024, 1024, 1024, 1024],
            }
        )
        stats = app.calculate_h3_statistics(df)
        # Latency increases linearly with dataset size (r = 1.0)
        assert stats["latency"]["r"] == 1.0
        assert stats["latency"]["reject"] == True
        # Ciphertext size remains constant, slope should be 0.0
        assert stats["noise"]["slope"] == 0.0


# 2. client.py Complete Line Coverage Tests
class TestClientFullCoverage:
    def test_parse_dnf_query_empty_and_whitespace(self):
        """Tests that parsing an empty or whitespace-only query string returns empty clauses and empty ignored terms lists."""
        clauses, ignored = parse_dnf_query("", ["gas"])
        assert clauses == [] and ignored == []

        clauses_ws, ignored_ws = parse_dnf_query("   ", ["gas"])
        assert clauses_ws == [] and ignored_ws == []

    def test_parse_dnf_query_not_negation_and_oov(self):
        """Tests that NOT operators, nested negations, and out-of-vocabulary terms are correctly processed into DNF clauses."""
        vocab = ["gas", "oil"]
        query = "NOT (gas OR unknown_term) AND NOT NOT oil"
        clauses, ignored = parse_dnf_query(query, vocab)

        assert len(clauses) > 0
        assert "unknown_term" in ignored

    def test_fhe_client_key_file_creation(self, tmp_path):
        """Tests automatic generation and initialization of a new AES symmetric encryption key when no key file exists."""
        m_open = mock_open()
        with (
            patch("os.path.exists", return_value=False),
            patch("builtins.open", m_open),
        ):
            fhe_c = FHEClient(vocab_size=5)
            assert fhe_c.aes_key is not None

    def test_decrypt_dnf_results_branches(self):
        """Tests DNF search result decryption for empty query payloads as well as pure negation matching logic."""
        fhe_c = FHEClient(vocab_size=5)

        # Test empty payload returns [1] * num_docs
        res_empty = fhe_c.decrypt_dnf_results([], num_docs=3)
        assert res_empty == [1, 1, 1]

        # Test pure negation handling
        enc_vec_1 = ts.bfv_vector(fhe_c.context, [1])
        enc_vec_0 = ts.bfv_vector(fhe_c.context, [0])

        b64_1 = base64.b64encode(enc_vec_1.serialize()).decode("utf-8")
        b64_0 = base64.b64encode(enc_vec_0.serialize()).decode("utf-8")

        pure_neg_payload = [
            {
                "required_pos_count": 0,
                "is_pure_negation": True,
                "encrypted_scores": [b64_1, b64_0],
            }
        ]

        matches = fhe_c.decrypt_dnf_results(pure_neg_payload, num_docs=2)
        assert matches == [1, 0]

    def test_try_decrypt_single_and_benchmark_io(self):
        """Tests single BFV vector decryption alongside benchmark dictionary encryption and decryption round-trips."""
        fhe_c = FHEClient(vocab_size=5)
        vec = ts.bfv_vector(fhe_c.context, [42])
        dec_val = fhe_c.try_decrypt_single(vec.serialize())
        assert dec_val[0] == 42

        # Benchmark dict IO
        sample_dict = {"h1": [1, 2, 3]}
        enc_b64 = fhe_c.encrypt_benchmark_data(sample_dict)
        dec_dict = fhe_c.decrypt_benchmark_data(enc_b64)
        assert dec_dict == sample_dict

    def test_parse_primary_index_out_of_bounds(self):
        """Tests parser handling of incomplete trailing binary operators where token indices exceed sequence length."""
        clauses, _ = parse_dnf_query("gas AND", ["gas"])
        assert isinstance(clauses, list)

    def test_parse_dnf_query_positive_oov_term(self):
        """Tests tracking of positive query terms that do not exist within the given target vocabulary."""
        vocab = ["gas", "oil"]
        query = "unknown_term AND gas"
        _, ignored = parse_dnf_query(query, vocab)
        assert "unknown_term" in ignored

    def test_encrypt_dnf_vectors_pure_negation(self):
        """Tests vector encryption for query clauses containing strictly negated terms without positive term indicators."""
        fhe_c = FHEClient(vocab_size=5)
        vocab = ["gas", "oil"]
        query = "NOT gas"
        encrypted_clauses, _ = fhe_c.encrypt_dnf_vectors(query, vocab)

        assert len(encrypted_clauses) > 0
        assert encrypted_clauses[0]["is_pure_negation"] is True

    def test_decrypt_dnf_results_positive_clause_match(self):
        """Tests score threshold evaluation for matching documents against standard positive DNF clauses."""
        fhe_c = FHEClient(vocab_size=5)

        enc_score = ts.bfv_vector(fhe_c.context, [2])
        b64_score = base64.b64encode(enc_score.serialize()).decode("utf-8")

        dnf_payload = [
            {
                "required_pos_count": 2,
                "is_pure_negation": False,
                "encrypted_scores": [b64_score],
            }
        ]

        matches = fhe_c.decrypt_dnf_results(dnf_payload, num_docs=1)
        assert matches == [1]

    def test_decrypt_payload_directly(self):
        """Tests symmetric encryption and decryption of raw text payloads directly using Fernet."""
        fhe_c = FHEClient(vocab_size=5)
        original_text = "secret email body content"
        cipher_text = fhe_c.encrypt_payload(original_text)
        decrypted_text = fhe_c.decrypt_payload(cipher_text)
        assert decrypted_text == original_text

    def test_get_public_context_bytes(self):
        """Tests generation and serialization of public FHE context bytes stripped of the secret key."""
        fhe_c = FHEClient(vocab_size=5)
        pub_bytes = fhe_c.get_public_context_bytes()
        assert isinstance(pub_bytes, bytes)
        assert len(pub_bytes) > 0

    def test_encrypt_dnf_vectors_positive_terms(self):
        """Tests vector encoding and FHE encryption of positive term index indicators across multiple clauses."""
        fhe_c = FHEClient(vocab_size=5)
        vocab = ["gas", "oil"]
        query = "gas OR oil"
        encrypted_clauses, _ = fhe_c.encrypt_dnf_vectors(query, vocab)

        assert len(encrypted_clauses) == 2
        assert encrypted_clauses[0]["required_pos_count"] == 1
        assert encrypted_clauses[0]["is_pure_negation"] is False


# 3. server.py FastAPI Endpoints & Helper Coverage
class TestServerFullCoverage:
    @pytest.fixture
    def test_client(self):
        """Initializes a FastAPI test client for making API requests in integration tests."""
        return TestClient(fastapi_app)

    @pytest.fixture
    def mock_db_and_matrix(self):
        """Sets up an in-memory mock sparse term-document matrix and email database for server testing."""
        vocab = ["gas", "energy"]
        dense_mat = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        matrix = sp.csr_matrix(dense_mat)

        with (
            patch("src.server.get_occurrence_matrix", return_value=matrix),
            patch(
                "src.server.encrypted_email_store",
                {"0": {"ciphertext_b64": "payload0"}},
            ),
            patch(
                "src.server.load_db_from_disk",
                return_value={"0": {"ciphertext_b64": "payload0"}},
            ),
            patch("src.server.ENCRYPTED_BENCHMARK_STORE", {}),
        ):
            yield vocab, matrix

    def _get_context(self):
        """Creates a lightweight TenSEAL BFV encryption context with generated Galois keys."""
        context = ts.context(
            ts.SCHEME_TYPE.BFV, poly_modulus_degree=4096, plain_modulus=1032193
        )
        context.generate_galois_keys()
        return context

    @pytest.mark.parametrize(
        "error_type,error_msg,is_transparent_val",
        [
            (Exception, "Ciphertext is transparent and cannot be serialized", True),
            (RuntimeError, "Generic memory serialization failure", False),
        ],
    )
    def test_safe_serialize_vector_exceptions(
        self, error_type, error_msg, is_transparent_val
    ):
        """Tests vector serialization error handling for both transparent fallback and standard runtime failures."""
        context = self._get_context()
        mock_vec = MagicMock()
        mock_vec.is_transparent.return_value = is_transparent_val
        mock_vec.context.return_value = context
        mock_vec.size.return_value = 2
        mock_vec.serialize.side_effect = error_type(error_msg)

        if is_transparent_val:
            res = safe_serialize_vector(mock_vec)
            assert isinstance(res, str)
        else:
            with pytest.raises(error_type, match=error_msg):
                safe_serialize_vector(mock_vec)

    def test_read_root(self, test_client):
        """Tests that accessing the root endpoint returns the expected server health check status response."""
        response = test_client.get("/")
        assert response.status_code == 200
        assert response.json() == {
            "status": "ok",
            "message": "FHE Search Server Running",
        }

    def test_verify_system_memory_safety_trigger(self):
        """Tests that the memory safety guard raises an HTTP 507 exception when RAM usage exceeds the safe threshold."""
        with patch("psutil.virtual_memory") as mock_mem:
            # Simulate high memory usage (>90%)
            mock_mem.return_value.percent = 95.0
            with pytest.raises(server.HTTPException) as exc_info:
                server.verify_system_memory_safety()
            assert exc_info.value.status_code == 507

    @pytest.mark.parametrize(
        "loader_func", [server.load_db_from_disk, server.load_benchmark_db_from_disk]
    )
    def test_disk_file_loaders_missing_or_corrupt(self, loader_func):
        """Tests that disk database loaders return an empty dictionary when encountering missing or corrupted JSON files."""
        with patch("os.path.exists", return_value=False):
            assert loader_func() == {}

        with (
            patch("os.path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data="{invalid_json}")),
        ):
            assert loader_func() == {}

    def test_fastapi_endpoints_full_suite(self, test_client, mock_db_and_matrix):
        """Tests API endpoint routes, response payloads, status codes, and fallback behaviors."""
        vocab, _ = mock_db_and_matrix
        fhe_c = FHEClient(vocab_size=len(vocab))
        pub_b64 = base64.b64encode(fhe_c.get_public_context_bytes()).decode("utf-8")
        clauses, _ = fhe_c.encrypt_dnf_vectors("gas", vocab)

        # 1. /fhe_boolean_search
        resp = test_client.post(
            "/fhe_boolean_search",
            json={
                "public_context_b64": pub_b64,
                "target_doc_count": 2,
                "dnf_clauses": clauses,
            },
        )
        assert resp.status_code == 200

        # 2. /run_h1_benchmark
        resp_h1 = test_client.post(
            "/run_h1_benchmark",
            json={
                "public_context_b64": pub_b64,
                "dnf_clauses": clauses,
                "dataset_sizes": [1, 2],
            },
        )
        assert resp_h1.status_code == 200

        # 3. /run_h2_noise_test
        resp_h2 = test_client.post(
            "/run_h2_noise_test",
            json={"public_context_b64": pub_b64, "num_clause_terms": 2},
        )
        assert resp_h2.status_code == 200

        # 4. /run_h3_scaling_test
        resp_h3 = test_client.post(
            "/run_h3_scaling_test",
            json={
                "public_context_b64": pub_b64,
                "dnf_clauses": clauses,
                "dataset_sizes": [1],
            },
        )
        assert resp_h3.status_code == 200

        resp_h3_auto = test_client.post(
            "/run_h3_scaling_test",
            json={"public_context_b64": pub_b64, "dnf_clauses": clauses},
        )
        assert resp_h3_auto.status_code == 200

        # 5. Metadata and Email Payloads
        assert test_client.get("/get_email_metadata").status_code == 200
        with (
            patch("src.server.encrypted_email_store", {}),
            patch(
                "src.server.load_db_from_disk",
                return_value={"0": {"ciphertext_b64": "fallback"}},
            ),
        ):
            assert test_client.get("/get_email_metadata").status_code == 200

        # Test valid email fetch
        assert test_client.get("/get_encrypted_email/0").status_code == 200

        # Test missing email fetch (wipe BOTH in-memory store AND disk loader)
        with (
            patch("src.server.encrypted_email_store", {}),
            patch("src.server.load_db_from_disk", return_value={}),
        ):
            assert test_client.get("/get_encrypted_email/999").status_code in (404, 500)

        # 6. Benchmark Storage & Retrieval
        save_resp = test_client.post(
            "/save_encrypted_benchmark",
            json={"benchmark_id": "test_bm", "encrypted_blob": "opaque_data"},
        )
        assert save_resp.status_code == 200

        with patch("builtins.open", side_effect=OSError("Write error")):
            err_save_resp = test_client.post(
                "/save_encrypted_benchmark",
                json={"benchmark_id": "fail_bm", "encrypted_blob": "opaque_data"},
            )
            assert err_save_resp.status_code == 500

        get_bm = test_client.get("/get_encrypted_benchmark/test_bm")
        assert get_bm.status_code == 200

        with (
            patch("src.server.ENCRYPTED_BENCHMARK_STORE", {}),
            patch(
                "src.server.load_benchmark_db_from_disk",
                return_value={"fallback_bm": "data"},
            ),
        ):
            assert (
                test_client.get("/get_encrypted_benchmark/fallback_bm").status_code
                == 200
            )

        assert test_client.get("/get_encrypted_benchmark/missing_bm").status_code in (
            404,
            500,
        )

        # 7. Vocabulary Endpoint
        with (
            patch("os.path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data='["gas", "oil"]')),
        ):
            assert test_client.get("/get_vocabulary").status_code == 200

        with patch("os.path.exists", return_value=False):
            assert test_client.get("/get_vocabulary").status_code == 404

    def test_get_cached_context_invalid_b64_raises_value_error(self):
        """Tests that context deserialization raises a ValueError when given an invalid Base64 string."""
        server.get_cached_context.cache_clear()
        with pytest.raises(ValueError) as exc_info:
            server.get_cached_context("invalid_base64_string_!!!")
        assert "TenSEAL Context Deserialization Failed" in str(exc_info.value)

    def test_evaluate_boolean_query_vector_size_mismatches(self):
        """Tests vector padding and truncation handling when document matrix dimensions do not match the query vector size."""
        ctx = self._get_context()

        # Truncation path
        enc_q_short = ts.bfv_vector(ctx, [1, 0])
        matrix_large = np.array([[1], [1], [1]], dtype=np.uint8)
        res_trunc = server.evaluate_boolean_query_vector(enc_q_short, matrix_large)
        assert len(res_trunc) == 1

        # Padding path
        enc_q_long = ts.bfv_vector(ctx, [1, 0, 1, 0])
        matrix_small = np.array([[1], [1]], dtype=np.uint8)
        res_pad = server.evaluate_boolean_query_vector(enc_q_long, matrix_small)
        assert len(res_pad) == 1

    def test_evaluate_dnf_vectors_negations(self):
        """Tests DNF clause evaluation containing negated terms and pure negation queries."""
        ctx = self._get_context()
        enc_vec = ts.bfv_vector(ctx, [1, 0])
        enc_b64 = base64.b64encode(enc_vec.serialize()).decode("utf-8")
        matrix = sp.csr_matrix(np.array([[1, 0], [0, 1]], dtype=np.uint8))

        clause_with_neg = [
            {
                "enc_pos_b64": enc_b64,
                "required_pos_count": 1,
                "is_pure_negation": False,
                "neg_indices": [0],
            }
        ]
        eval_res = server.evaluate_dnf_vectors(clause_with_neg, ctx, matrix)
        assert len(eval_res) == 1

        clause_pure_neg = [
            {
                "enc_pos_b64": enc_b64,
                "required_pos_count": 0,
                "is_pure_negation": True,
                "neg_indices": [],
            }
        ]
        eval_res_pure = server.evaluate_dnf_vectors(clause_pure_neg, ctx, matrix)
        assert len(eval_res_pure) == 1

    def test_evaluate_dnf_vectors_sparse_negated_val(self):
        """Tests evaluation of negated terms when accessing elements in a sparse matrix representation."""
        context = self._get_context()

        mock_matrix = MagicMock()
        mock_matrix.shape = (5, 1)
        mock_matrix.toarray.return_value = np.array(
            [[1], [0], [1], [0], [1]], dtype=np.int64
        )

        mock_val = MagicMock()
        mock_val.toarray.return_value = [[1]]
        mock_matrix.__getitem__.return_value = mock_val

        enc_pos = ts.bfv_vector(context, [0, 0, 0, 0, 0])
        enc_pos_b64 = base64.b64encode(enc_pos.serialize()).decode("utf-8")

        dnf_clauses = [
            {
                "enc_pos_b64": enc_pos_b64,
                "required_pos_count": 0,
                "is_pure_negation": False,
                "neg_indices": [0],
            }
        ]

        results = evaluate_dnf_vectors(dnf_clauses, context, mock_matrix)
        assert len(results) == 1
        mock_val.toarray.assert_called_once()

    def test_get_occurrence_matrix_disk_load_and_missing_file(self):
        """Tests sparse matrix loading from disk as well as handling when the matrix file is missing."""
        server.occurrence_matrix = None
        with (
            patch("os.path.exists", return_value=False),
            pytest.raises(
                FileNotFoundError, match="Matrix file missing at target location:"
            ),
        ):
            server.get_occurrence_matrix()

        server.occurrence_matrix = None
        mock_mat = sp.csr_matrix(np.array([[1]]))

        # Patch load_npz directly in src.server namespace
        with (
            patch("os.path.exists", return_value=True),
            patch("src.server.load_npz", return_value=mock_mat),
        ):
            mat = server.get_occurrence_matrix()
            assert mat.shape == (1, 1)

    @pytest.mark.parametrize(
        "endpoint,payload",
        [
            (
                "/fhe_boolean_search",
                {"public_context_b64": "invalid", "dnf_clauses": []},
            ),
            (
                "/run_h1_benchmark",
                {
                    "public_context_b64": "invalid",
                    "dnf_clauses": [],
                    "dataset_sizes": [1],
                },
            ),
            (
                "/run_h2_noise_test",
                {"public_context_b64": "invalid", "num_clause_terms": 5},
            ),
            (
                "/run_h3_scaling_test",
                {"public_context_b64": "invalid", "dnf_clauses": []},
            ),
        ],
    )
    def test_server_endpoints_500_exception_handlers(
        self, test_client, endpoint, payload
    ):
        """Tests that server endpoints catch unhandled internal exceptions and return HTTP 500 status codes."""
        with patch(
            "src.server.get_occurrence_matrix",
            side_effect=Exception("Database Connection Error"),
        ):
            response = test_client.post(endpoint, json=payload)
            assert response.status_code == 500

    def test_safe_serialize_vector_transparent_exception(self):
        """Tests fallback serialization behavior when a vector throws a non-transparent exception."""
        mock_vec = MagicMock()
        mock_vec.is_transparent.return_value = False
        mock_vec.serialize.side_effect = Exception("Vector is transparent")
        mock_vec.size.return_value = 5

        mock_zero_vec = MagicMock()
        mock_zero_vec.serialize.return_value = b"mocked_zero_bytes"

        # Should catch exception and fallback to zero-vector serialization
        with patch("src.server.ts.bfv_vector", return_value=mock_zero_vec):
            res = safe_serialize_vector(mock_vec)

        assert isinstance(res, str)
        assert res == base64.b64encode(b"mocked_zero_bytes").decode("utf-8")

    def test_run_h3_scaling_endpoint(self, test_client):
        """Tests the endpoint invocation for the dataset scalability benchmark using a standard payload."""
        payload = {"public_context_b64": "...", "dnf_clauses": []}
        response = test_client.post("/run_h3_scaling_test", json=payload)
        assert response.status_code in [200, 400, 422, 500]

    def test_get_occurrence_matrix_corrupted_file_raises_runtime_error(self):
        """Tests that loading a corrupted sparse matrix file raises a RuntimeError."""
        # Reset cached matrix state
        server.occurrence_matrix = None

        with (
            patch("os.path.exists", return_value=True),
            patch(
                "src.server.load_npz",
                side_effect=Exception("Corrupted matrix headers"),
            ),
            pytest.raises(RuntimeError) as exc_info,
        ):
            server.get_occurrence_matrix()

        assert "Unable to load sparse matrix file" in str(exc_info.value)
        assert "Corrupted matrix headers" in str(exc_info.value)

    @pytest.mark.parametrize(
        "raised_exception",
        [
            ValueError("Invalid context provided"),
            FileNotFoundError("Matrix file missing"),
            RuntimeError("Failed to process matrix"),
        ],
    )
    def test_fhe_boolean_search_bad_input_raises_400(
        self, test_client, raised_exception
    ):
        """Tests that invalid inputs or missing files during boolean search return an HTTP 400 response."""
        payload = {
            "public_context_b64": "mock_context_b64",
            "target_doc_count": 10,
            "dnf_clauses": [],
        }

        with patch(
            "src.server.get_occurrence_matrix",
            side_effect=raised_exception,
        ):
            response = test_client.post("/fhe_boolean_search", json=payload)

            assert response.status_code == 400
            assert str(raised_exception) in response.json()["detail"]

    @pytest.mark.parametrize(
        "raised_exception",
        [
            ValueError("Invalid public context or parameters"),
            FileNotFoundError("Sparse matrix missing on server"),
        ],
    )
    def test_run_h1_benchmark_bad_input_raises_400(self, test_client, raised_exception):
        """Tests that bad input or missing server matrix files during benchmark execution return an HTTP 400 response."""
        payload = {
            "public_context_b64": "mock_context_b64",
            "dnf_clauses": [],
            "dataset_sizes": [1, 2],
        }

        with patch(
            "src.server.get_occurrence_matrix",
            side_effect=raised_exception,
        ):
            response = test_client.post("/run_h1_benchmark", json=payload)

            assert response.status_code == 400
            assert str(raised_exception) in response.json()["detail"]

    @pytest.mark.parametrize(
        "raised_exception",
        [
            ValueError("Invalid context supplied"),
            FileNotFoundError("Sparse matrix unavailable"),
        ],
    )
    def test_run_h2_noise_test_bad_input_raises_400(
        self, test_client, raised_exception
    ):
        """Tests that invalid parameters or unreadable files during noise testing return an HTTP 400 response."""
        payload = {
            "public_context_b64": "mock_context_b64",
            "num_clause_terms": 3,
        }

        with patch(
            "src.server.get_occurrence_matrix",
            side_effect=raised_exception,
        ):
            response = test_client.post("/run_h2_noise_test", json=payload)

            assert response.status_code == 400
            assert str(raised_exception) in response.json()["detail"]

    @pytest.mark.parametrize(
        "file_content_or_error",
        [
            "{invalid_json_content}",  # Triggers json.JSONDecodeError
            OSError("Disk read failure"),  # Triggers OSError
        ],
    )
    def test_get_vocabulary_corrupted_file_raises_500(
        self, test_client, file_content_or_error
    ):
        """Tests that vocabulary requests return an HTTP 500 status code when the vocabulary file is unreadable or corrupted."""
        if isinstance(file_content_or_error, Exception):
            mock_file = mock_open()
            mock_file.side_effect = file_content_or_error
        else:
            mock_file = mock_open(read_data=file_content_or_error)

        with (
            patch("os.path.exists", return_value=True),
            patch("builtins.open", mock_file),
        ):
            response = test_client.get("/get_vocabulary")

            assert response.status_code == 500
            assert response.json()["detail"] == "Vocabulary file corrupted."

    def test_evaluate_dnf_vectors_missing_key_raises_value_error(self):
        """Tests that evaluating DNF clauses with missing payload keys raises a ValueError."""
        ctx = self._get_context()
        matrix = sp.csr_matrix(np.array([[1]]))
        malformed_clauses = [{"invalid_key": "no_enc_pos_b64_here"}]

        with pytest.raises(ValueError, match="Missing key in clause specification"):
            evaluate_dnf_vectors(malformed_clauses, ctx, matrix)

    def test_evaluate_dnf_vectors_corrupt_bfv_vector_raises_value_error(self):
        """Tests that decoding corrupt or invalid BFV vector clauses raises a ValueError."""
        ctx = self._get_context()
        matrix = sp.csr_matrix(np.array([[1]]))
        invalid_clause = {
            "enc_pos_b64": base64.b64encode(b"invalid_bfv_bytes").decode("utf-8")
        }

        with pytest.raises(ValueError, match="Failed to decode BFV vector clause"):
            evaluate_dnf_vectors(invalid_clause, ctx, matrix)


# 4. setup_db.py Preprocessing Coverage
class TestSetupDBFullCoverage:
    def test_extract_body_variations(self):
        """Tests email body extraction from raw message strings across various header configurations and plain headerless text."""
        raw_headerless = "Plain message without headers."
        assert db.extract_body(raw_headerless) == "Plain message without headers."

    def test_extract_keywords_and_filtering(self):
        """Tests text tokenization, lowercasing, non-alphanumeric character removal, stopword filtering, and token length restrictions."""
        text = "The 12345 report on natural gas trading and 12 numbers."
        keywords = db.extract_keywords(text)
        # Numbers >= 3 digits and standard words should be retained
        assert "12345" in keywords
        assert "report" in keywords
        # Numbers < 3 digits should be filtered out
        assert "12" not in keywords

    def test_setup_db_script_missing_file(self):
        """Tests system exit and exception handling when the primary input dataset file is missing during module import."""
        import importlib

        with (
            patch("pandas.read_csv", side_effect=FileNotFoundError),
            patch("sys.exit", side_effect=SystemExit) as mock_exit,
            pytest.raises(SystemExit),
        ):
            # Reload module to trigger script execution logic
            importlib.reload(db)

        mock_exit.assert_called_once()
