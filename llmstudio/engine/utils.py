import re


def is_valid_endpoint_name(name: str) -> bool:
    """
    Check whether a string contains any URL reserved characters, spaces, or characters other
    than alphanumeric, underscore, hyphen, and dot.

    Returns True if the string doesn't contain any of these characters.
    """
    pattern = r"[^\w\.-]"
    return re.search(pattern, name) is None


def validate_provider_config(config, api_key):
    """
    Validate and/or initialize a provider configuration based on input parameters.

    Parameters:
    - config (dict or None): Configuration dictionary for the provider. Can be None.
    - api_key (str or None): API key for the provider. Can be None.

    Returns:
    dict: The modified or validated configuration dictionary.

    Raises:
    - ValueError: If both `config` and `api_key` are None.
    """
    if not (config or api_key):
        raise ValueError(f"Config was not specified neither an api_key was provided.")
    if config is None:
        config = {}
    if api_key is not None:
        config.setdefault("api_key", api_key)
    return config


def check_configuration_route_name_collisions(config):
    """
    Checks for duplicate route names in the given configuration.

    Parameters:
        config (dict): The configuration dictionary containing a list of routes.
                    Each route should be a dictionary with a 'name' key.

    Returns:
        None: Returns None if there are no duplicates.

    Raises:
        ValueError: If there are duplicate route names found in the configuration.
    """
    if len(config["routes"]) < 2:
        return
    names = [route["name"] for route in config["routes"]]
    if len(names) != len(set(names)):
        raise ValueError(
            "Duplicate names found in route configurations. Please remove the duplicate route "
            "name from the configuration to ensure that route endpoints are created properly."
        )
