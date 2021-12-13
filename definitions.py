"""This file contains global definitions"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

RUN_DIR = ROOT_DIR / "runs"
LOG_ROOT = "recommender"
LOG_DIR = ROOT_DIR / "logs"
LOG_FILE_NAME = LOG_DIR / "recommender_logs.log"
MIGRATIONS_DIR = ROOT_DIR / "recommender" / "migrate" / "migrations"
