# -*- coding: utf-8; -*-
"""
Copyright (c) 2018 Rolf Hempel, rolf6419@gmx.de

This file is part of the PlanetarySystemStacker tool (PSS).
https://github.com/Rolf-Hempel/PlanetarySystemStacker

PSS is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with PSS.  If not, see <http://www.gnu.org/licenses/>.

"""

class Error(Exception):
    """Base class for exceptions in this module.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message

class TypeError(Error):
    """Exception raised for type errors during input.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        super(TypeError, self).__init__(message)

class ShapeError(Error):
    """Exception raised if image shapes do not match during input.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        super(ShapeError, self).__init__(message)

class NotSupportedError(Error):
    """Exception raised if a method is called which is not implemented yet.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        super(NotSupportedError, self).__init__(message)

class ArgumentError(Error):
    """Exception raised if a method is called which an invalid argument.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        super(ArgumentError, self).__init__(message)

class WrongOrderingError(Error):
    """Exception raised if a method is called before required input is available.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        super(WrongOrderingError, self).__init__(message)

class InternalError(Error):
    """Exception raised if an internal error occurred which should never happen.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        super(InternalError, self).__init__(message)

class DivideByZeroError(Error):
    """Exception raised if an attempt is made to divide by zero.

        Attributes:
            message -- explanation of the error
        """

    def __init__(self, message):
        super(DivideByZeroError, self).__init__(message)

class IncompatibleVersionsError(Error):
    """Exception raised if data and program versions do not match.

        Attributes:
            message -- explanation of the error
        """

    def __init__(self, message):
        super(IncompatibleVersionsError, self).__init__(message)
