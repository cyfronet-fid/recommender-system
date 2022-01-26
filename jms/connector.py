"""Mini framework for JMS interaction"""
import stomp
from logger_config import get_logger

logger = get_logger(__name__)


class JMSConnector:
    """Establish a JMS connection using the stomp protocol"""

    def __init__(self, host, port, login, password):
        self.host = host
        self.port = port
        self.login = login
        self.password = password
        self.connection = None

    def connect(self):
        """Establish a connection"""
        self.connection = stomp.Connection([(self.host, self.port)])
        self.connection.connect(self.login, self.password, wait=True)

    def check_connection(self):
        """Check the status of connection"""
        if self.connection:
            logger.info(
                "The status of your connection: %s", self.connection.is_connected()
            )
            return self.connection.is_connected()

        return False

    def disconnect(self):
        """Disconnect"""
        if self.connection.is_connected():
            self.connection.disconnect()
        else:
            logger.debug("There is no established connection. Aborting...")
