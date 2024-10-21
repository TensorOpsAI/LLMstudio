class CustomError(Exception):
    """Base class for all custom exceptions"""


class ProviderError(CustomError):
    """Raised when there is a validation error"""
