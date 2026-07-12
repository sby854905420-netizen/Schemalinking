from __future__ import annotations

import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PredictionPathInvariantTests(unittest.TestCase):
    def test_prediction_entrypoints_do_not_reference_logs_storage(self):
        entrypoints = (
            "Baseline/baseline_database_retrieval.py",
            "Baseline/baseline_schema_linking.py",
            "Baseline/rag_column_retrieval.py",
            "Run/global_coarse_retrieval.py",
            "Run/table_to_column.py",
            "Run/sql_generator.py",
            "Run/one_shot_sql_generator.py",
        )
        forbidden = (
            "LOGS_ROOT",
            "--logs-dir",
            "resolve_input_path",
            "resolve_output_path",
            "find_latest_schema_result_file",
            "atomic_write_json",
        )
        for relative_path in entrypoints:
            source = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
            for marker in forbidden:
                self.assertNotIn(marker, source, f"{relative_path} contains {marker}")

    def test_logs_root_is_used_only_for_logging_and_input_rejection(self):
        references = []
        for directory in ("Baseline", "Run", "Utils"):
            for path in (PROJECT_ROOT / directory).rglob("*.py"):
                if "LOGS_ROOT" in path.read_text(encoding="utf-8"):
                    references.append(path.relative_to(PROJECT_ROOT).as_posix())
        self.assertEqual(
            references,
            ["Utils/artifact_paths.py", "Utils/logging_utils.py"],
        )

    def test_logs_prediction_inputs_and_outputs_are_rejected(self):
        from config import LOGS_ROOT
        from Utils.artifact_paths import (
            reject_logs_prediction_input,
            require_results_output,
        )

        candidate = LOGS_ROOT / "prediction.json"
        with self.assertRaises(ValueError):
            reject_logs_prediction_input(candidate)
        with self.assertRaises(ValueError):
            require_results_output(candidate)


if __name__ == "__main__":
    unittest.main()
