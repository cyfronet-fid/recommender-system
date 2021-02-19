"""Api namespace definition for use by all schema models"""

from flask_restx import Namespace, fields

api = Namespace("schemas")


class NullableString(fields.String):
    """Custom field to handle string or null"""

    __schema_type__ = ["string", "null"]
    __schema_example__ = "nullable string"
