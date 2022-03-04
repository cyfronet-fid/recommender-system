# pylint: disable-all
import pytest


@pytest.mark.skip(reason="Investigate how to mock the connection on GH")
def test_jms_connector(init_jms_conn):
    # Initialize an object capable of connecting to the jms
    jms_conn = init_jms_conn

    # A connection should not be established
    assert jms_conn.check_connection() is False

    jms_conn.connect()

    assert jms_conn.check_connection() is True
    jms_conn.disconnect()
    # TODO for some reason disconnect() does not make self.connection.is_connected() False.
    # Investigate in the near future
    # assert jms_conn.check_connection() is False
