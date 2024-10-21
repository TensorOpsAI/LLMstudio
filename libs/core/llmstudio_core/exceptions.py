class CustomError(Exception):
    """Base class for all custom exceptions"""

    pass


class ProviderError(CustomError):
    """Raised when there is a validation error"""

    pass
