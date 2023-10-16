import json
import re

def append_log(data):
    path = "/api/logs/execution_logs.jsonl"

    with open(path, "a") as file:
        file.write(json.dumps(data) + "\n")


def is_valid_endpoint_name(name: str) -> bool:
    """
    Check whether a string contains any URL reserved characters, spaces, or characters other
    than alphanumeric, underscore, hyphen, and dot.

    Returns True if the string doesn't contain any of these characters.
    """
    pattern = r'[^\w\.-]'
    return re.search(pattern, name) is None

def validate_provider_config(config, api_key):
    if not (config or api_key):
        raise ValueError(
            f"Config was not specified neither an api_key was provided."
        )
    if config is None:
        config = {}
    if api_key is not None:
        config.setdefault('api_key', api_key)
    return config