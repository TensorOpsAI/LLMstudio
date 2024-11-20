import json

from sqlalchemy import Text
from sqlalchemy.types import TypeDecorator


class JSONEncodedDict(TypeDecorator):
    """JSON-encoded dictionary for compatibility with BigQuery."""

    impl = Text

    def process_bind_param(self, value, dialect):
        if value is not None:
            return json.dumps(value)
        return None

    def process_result_value(self, value, dialect):
        if value is not None:
            return json.loads(value)
        return None
