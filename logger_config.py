# pylint: disable=line-too-long, too-few-public-methods

"""File containing logger configuration"""

import os
import logging
from logging.config import dictConfig
from definitions import LOG_FILE_NAME, LOG_ROOT

LOG_SIMPLE_FORMAT = "%(asctime)s | %(name)-s | %(levelname)-s | %(message)s"
LOG_DETAILED_FORMAT = "%(asctime)s | %(name)-s | %(levelname)-s | %(message)s | %(filename)s/%(funcName)s [%(lineno)d]"


def apply_logging_config():
    """Apply predefined logging configuration"""
    dictConfig(logging_config)


def get_logger(logger_name: str):
    """Get a logger using its name"""
    if not os.environ["FLASK_ENV"] == "testing":
        apply_logging_config()

    return logging.getLogger(logger_name)


class InfoFilter(logging.Filter):
    """Filter events above INFO level"""

    def filter(self, record):
        return record.levelname == "INFO"


logging_config = dict(
    version=1,
    formatters={
        "file": {"format": LOG_DETAILED_FORMAT},
        "detailed_console": {
            "()": "coloredlogs.ColoredFormatter",
            "format": LOG_DETAILED_FORMAT,
        },
        "simple_console": {
            "()": "coloredlogs.ColoredFormatter",
            "format": LOG_SIMPLE_FORMAT,
        },
    },
    filters={
        "info_filter": {
            "()": InfoFilter,
        }
    },
    handlers={
        "detailed_console": {
            "level": logging.WARNING,
            "class": "logging.StreamHandler",
            "formatter": "detailed_console",
        },
        "only_info_console": {
            "level": logging.INFO,
            "class": "logging.StreamHandler",
            "formatter": "simple_console",
            "filters": ["info_filter"],
        },
        "server_console": {
            "level": logging.INFO,
            "class": "logging.StreamHandler",
            "formatter": "simple_console",
        },
        "file": {
            "level": logging.WARNING,
            "class": "logging.handlers.RotatingFileHandler",
            "filename": LOG_FILE_NAME,
            "formatter": "file",
            "maxBytes": 1024 * 1024 * 100,  # ~100 MB
            "backupCount": 10,
        },
    },
    loggers={
        LOG_ROOT: {
            "handlers": ["only_info_console", "detailed_console", "file"],
            "level": logging.INFO,
            "propagate": False,
        },
        "": {  # Root logger
            "handlers": ["server_console", "file"],
            "level": logging.INFO,
            "propagate": False,
        },
        "werkzeug": {  # Flask
            "handlers": ["server_console", "file"],
            "level": logging.INFO,
            "propagate": False,
        },
    },
)
