# pylint: disable-all

"""Pytest utilty fixtures"""
import pytest
from mongoengine import connect, disconnect


@pytest.fixture(scope="function")
def mongo():
    conn = connect(db="test", host="mongomock://localhost")
    yield conn
    conn.drop_database("test")
    disconnect()
