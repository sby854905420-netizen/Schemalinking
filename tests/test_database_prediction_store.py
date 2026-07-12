from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from Utils.database_prediction_store import (
    DatabasePredictionValidationError,
    build_database_prediction,
    build_database_prediction_path,
    database_predictions_to_native_records,
    initialize_database_prediction_file,
    load_database_prediction_records,
    resolve_database_prediction_input,
    upsert_database_prediction,
    validate_database_prediction_file,
)
from Utils.json_utils import atomic_write_json


class DatabasePredictionStoreTests(unittest.TestCase):
    def test_input_resolution_uses_only_explicit_or_results_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unified = build_database_prediction_path(
                "baseline", "MMQA", "model", results_root=root / "results"
            )
            initialize_database_prediction_file(
                unified,
                dataset_name="MMQA",
                method="baseline",
                model_name="model",
            )
            self.assertEqual(
                resolve_database_prediction_input(
                    explicit_path=None,
                    method="baseline",
                    dataset_name="MMQA",
                    model_name="model",
                    results_root=root / "results",
                ),
                unified,
            )
            explicit = root / "custom.json"
            self.assertEqual(
                resolve_database_prediction_input(
                    explicit_path=explicit,
                    method="baseline",
                    dataset_name="MMQA",
                    model_name="model",
                    results_root=root / "results",
                ),
                explicit,
            )

    def test_missing_results_prediction_does_not_fall_back_to_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            logs_file = root / "Logs" / "legacy.json"
            atomic_write_json(logs_file, [{"id": "1", "predict_db_id": "db"}])
            with self.assertRaises(FileNotFoundError):
                resolve_database_prediction_input(
                    explicit_path=None,
                    method="baseline",
                    dataset_name="MMQA",
                    model_name="model",
                    results_root=root / "results",
                )

    def test_path_and_atomic_upsert_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = build_database_prediction_path(
                "global_coarse_rerank", "MMQA", "org/model", results_root=root
            )
            self.assertEqual(
                path.relative_to(root).as_posix(),
                "db/global_coarse_rerank/MMQA/org__model/prediction.json",
            )
            initialize_database_prediction_file(
                path,
                dataset_name="MMQA",
                method="global_coarse_rerank",
                model_name="org/model",
            )
            upsert_database_prediction(
                path,
                build_database_prediction(
                    sample_id=1,
                    question="question",
                    predicted_db_id="db_one",
                    efficiency={
                        "sample_elapsed_seconds": 1.25,
                        "llm_total_tokens": 7,
                    },
                ),
            )
            payload = validate_database_prediction_file(path)
            self.assertEqual(payload["predictions"][0]["predicted_db_id"], "db_one")
            native = database_predictions_to_native_records(payload)
            self.assertEqual(native[0]["predict_db_id"], "db_one")
            self.assertEqual(load_database_prediction_records(path), native)

    def test_legacy_list_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.json"
            atomic_write_json(path, [{"id": "1", "predict_db_id": "db"}])
            with self.assertRaises(DatabasePredictionValidationError):
                load_database_prediction_records(path)

if __name__ == "__main__":
    unittest.main()
