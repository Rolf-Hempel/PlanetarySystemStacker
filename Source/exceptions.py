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

class WrongOrderingError(Error):
    """Exception raised if a method is called before required input is available.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message
