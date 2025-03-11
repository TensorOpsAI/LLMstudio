class CustomError(Exception):
    """Base class for all custom exceptions"""


class ProviderError(CustomError):
    """Raised when there is a validation error on provider"""


class AgentError(CustomError):
    """Raised when there is a validation error on agent"""


class InputMessageError(CustomError):
    """Raised when there is a validation error on agent message"""
