class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class TypeError(Error):
    """Exception raised for type errors during input.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message

class ShapeError(Error):
    """Exception raised if image shapes do not match during input.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message

class NotSupportedError(Error):
    """Exception raised if a method is called which is not implemented yet.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message

class ArgumentError(Error):
    """Exception raised if a method is called which an invalid argument.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message
