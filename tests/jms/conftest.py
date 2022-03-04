# pylint: disable-all

"""Fixtures used by jms tests"""
import pytest
import os

from jms.connector import JMSConnector


@pytest.fixture
def get_env_variables():
    """Get environmental variables"""
    host = os.environ.get("JMS_HOST")
    port = os.environ.get("JMS_PORT")
    login = os.environ.get("JMS_LOGIN")
    password = os.environ.get("JMS_PASSWORD")

    return host, port, login, password


@pytest.fixture
def init_jms_conn(get_env_variables):
    """Initialize JMS connection"""
    init_jms = JMSConnector(*get_env_variables)

    return init_jms


@pytest.fixture
def establish_jms_conn(init_jms_conn):
    """Establish JMS connection"""
    jms_conn = init_jms_conn
    jms_conn.connect()

    return jms_conn
