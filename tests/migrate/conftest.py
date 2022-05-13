# pylint: disable-all
"""Fixtures for migrations testing"""
import pytest

from definitions import ROOT_DIR


@pytest.fixture
def mock_migrations_dir(monkeypatch):
    """Mock migrations directory"""
    mock_dir = ROOT_DIR / "tests" / "migrate" / "mock_migrations"
    monkeypatch.setattr("recommender.migrate.utils.MIGRATIONS_DIR", mock_dir)
    return mock_dir
