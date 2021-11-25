"""Exposing all flask CLI commands"""

from .db import db_command
from .migrate import migrate_command
from .train import train_command
