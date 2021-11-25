"""This file contains global definitions"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
LOG_DIR = ROOT_DIR / "runs"
MIGRATIONS_DIR = ROOT_DIR / "recommender" / "migrate" / "migrations"
