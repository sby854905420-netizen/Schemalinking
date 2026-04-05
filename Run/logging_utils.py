from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


def setup_task_logger(task_name: str, result_path: Path) -> tuple[logging.Logger, Path]:
    log_file_path = result_path.with_suffix(".log")
    logger_name = f"{task_name}.{result_path.stem}"
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
