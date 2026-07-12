from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from config import LOGS_ROOT
from Utils.prediction_store import safe_path_component


def build_run_log_path(task_name: str, dataset_name: str, model_name: str) -> Path:
    """Return a timestamped text-log path; Logs never contain prediction JSON."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    directory = LOGS_ROOT / safe_path_component(model_name) / safe_path_component(task_name)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{safe_path_component(task_name)}_{safe_path_component(dataset_name)}_{run_id}.log"


def setup_task_logger(task_name: str, log_file_path: Path) -> tuple[logging.Logger, Path]:
    """Create a console/file logger for an explicit runtime ``.log`` path."""
    if log_file_path.suffix != ".log":
        raise ValueError(f"Runtime log path must end in .log: {log_file_path}")
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    logger_name = f"{task_name}.{log_file_path.stem}"
    logger = logging.getLogger(logger_name)

    if logger.handlers:
        return logger, log_file_path

    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger, log_file_path


def log_run_configuration(
    logger: logging.Logger,
    *,
    task_name: str,
    dataset_name: str,
    data_count: int,
    model_name: str,
    provider: str,
    result_path: Path,
    extra_fields: dict[str, Any] | None = None,
) -> None:
    logger.info("Task: %s", task_name)
    logger.info("Dataset: %s", dataset_name)
    logger.info("Loaded data count: %s", data_count)
    logger.info("Model: %s", model_name)
    logger.info("Provider: %s", provider)
    logger.info("Result path: %s", result_path)

    for key, value in (extra_fields or {}).items():
        logger.info("%s: %s", key, value)
