"""
LexGuard — Centralized Pipeline Logger
=======================================
Provides structured logging to console + file, with helpers for
recording hyperparameters and execution metrics.

Usage:
    from lexguard_logger import get_logger, log_metrics, write_run_manifest
    logger = get_logger(__name__)
    logger.info("Starting ingestion …")
"""

import os
import json
import time
import logging
import datetime
from pathlib import Path

# ──────────────────────────────────────────────
# Directory setup
# ──────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent
_LOG_DIR = _PROJECT_ROOT / "logs"
_ARTIFACT_DIR = _PROJECT_ROOT / "artifacts"
_LOG_DIR.mkdir(exist_ok=True)
_ARTIFACT_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────
# Logger factory
# ──────────────────────────────────────────────
_LOG_FILE = _LOG_DIR / "pipeline_run.log"

_formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

# File handler — appends per-run
_fh = logging.FileHandler(_LOG_FILE, mode="a", encoding="utf-8")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(_formatter)

# Console handler
_ch = logging.StreamHandler()
_ch.setLevel(logging.INFO)
_ch.setFormatter(_formatter)


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger that writes to console and log file."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        logger.addHandler(_fh)
        logger.addHandler(_ch)
    return logger


# ──────────────────────────────────────────────
# Hyperparameter logging
# ──────────────────────────────────────────────
def log_hyperparams(logger: logging.Logger, params: dict) -> None:
    """Log a dict of hyperparameters in a readable block."""
    logger.info("=" * 50)
    logger.info("HYPERPARAMETERS")
    logger.info("=" * 50)
    for key, value in params.items():
        logger.info(f"  {key:30s} = {value}")
    logger.info("=" * 50)


# ──────────────────────────────────────────────
# Execution metrics
# ──────────────────────────────────────────────
_metrics_store: list[dict] = []


def log_metrics(logger: logging.Logger, **kwargs) -> None:
    """Record arbitrary key-value metrics for the current run.

    Example:
        log_metrics(logger, phase="ingestion", chunks=142, elapsed_s=3.7)
    """
    entry = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        **kwargs,
    }
    _metrics_store.append(entry)
    logger.info(f"METRIC | {json.dumps(kwargs)}")


# ──────────────────────────────────────────────
# Run manifest (written at end of pipeline)
# ──────────────────────────────────────────────
def write_run_manifest(hyperparams: dict | None = None) -> Path:
    """Write a JSON manifest summarising the pipeline run to artifacts/.

    Returns the path to the written file.
    """
    manifest = {
        "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hyperparams": hyperparams or {},
        "metrics": _metrics_store,
    }
    out_path = _ARTIFACT_DIR / "run_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_path
