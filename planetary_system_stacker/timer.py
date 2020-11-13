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

from time import time, sleep
from collections import OrderedDict

from exceptions import ArgumentError
from miscellaneous import Miscellaneous


class timer(object):
    """
    Measure execution times of code sections. Several timers can be used at the same time. Timers
    can be created, incremented, and deleted.
    """

    def __init__(self):
        """
        Initialize the timer object.
        """

        self.counters = OrderedDict()

    def create(self, name):
        """
        Create a named timer, and start it.

        :param name: Name of the timer
        :return: -
        """

        if name in self.counters.keys():
            raise ArgumentError("Attempt to initialize timer with existing name")
        else:
            self.counters[name] = [0., time()]

    def create_no_check(self, name):
        """
        If a named timer does not exist, create one. If it exists, reset and start it.

        :param name: Name of the timer
        :return: -
        """

        self.counters[name] = [0., time()]

    def delete(self, name):
        """
        Delete a named timer.

        :param name: Name of the timer
        :return: -
        """

        if name not in self.counters.keys():
            raise ArgumentError("Attempt to delete timer with undefined name")
        else:
            del self.counters[name]

    def exists(self, name):
        """
        Find out if a named timer with a given name already exists.

        :param name: Name of the timer
        :return: True, if the timer exists; otherwise False.
        """

        return name in self.counters.keys()

    def start(self, name):
        """
        Start a timer.

        :param name: Name of the timer
        :return: -
        """

        if name not in self.counters.keys():
            raise ArgumentError("Attempt to start timer with undefined name")
        else:
            self.counters[name][1] = time()

    def stop(self, name):
        """
        Stop a timer.

        :param name: Name of the timer
        :return: Elapsed time accumulated during start/stop intervals of this timer so far.
        """

        if name not in self.counters.keys():
            raise ArgumentError("Attempt to stop timer with undefined name")
        else:
            self.counters[name][0] += time() - self.counters[name][1]
            return self.counters[name][0]

    def read(self, name):
        """
        Read out a timer.

        :param name: Name of the timer
        :return: Elapsed time accumulated during start/stop intervals of this timer so far.
        """

        if name not in self.counters.keys():
            raise ArgumentError("Attempt to read out timer with undefined name")
        else:
            return self.counters[name][0]

    def reset(self, name):
        """
        Reset a timer to zero without deleting it.

        :param name: Name of the timer.
        :return: -
        """

        if name not in self.counters.keys():
            raise ArgumentError("Attempt to reset timer with undefined name")
        else:
            self.counters[name][0] = 0.

    def print(self):
        print("--------------------------------------------------\nStatus of time counters:")
        for name in self.counters.keys():

            print("{0:40} {1:8.3f}".format(name, self.counters[name][0]))
        print("--------------------------------------------------")

    def protocol(self, logfile):
        Miscellaneous.protocol("", logfile, precede_with_timestamp=False)
        Miscellaneous.protocol("           --------------------------------------------------\n"
                               "           Status of time counters:", logfile,
                               precede_with_timestamp=False)
        for name in self.counters.keys():
            Miscellaneous.protocol(
                "           {0:40} {1:8.3f}".format(name, self.counters[name][0]), logfile,
                precede_with_timestamp=False)
        Miscellaneous.protocol("           --------------------------------------------------\n",
                               logfile, precede_with_timestamp=False)


if __name__ == "__main__":
    my_timer = timer()
    my_timer.create('Second')
    my_timer.create('First')
    try:
        my_timer.create('First')
    except ArgumentError as e:
        print(e.message)

    my_timer.start('First')
    sleep(0.1)
    print("First timer: " + str(my_timer.stop('First')))

    my_timer.start('First')
    sleep(0.5)
    print("First timer: " + str(my_timer.stop('First')))

    start = time()
    sleep(0.1)
    print("Elapsed time (without timer): " + str(time() - start))

    my_timer.start('Second')
    sleep(0.1)
    print("Second timer: " + str(my_timer.stop('Second')))

    my_timer.reset('First')
    my_timer.start('First')
    sleep(0.5)
    print("First timer: " + str(my_timer.stop('First')))

    my_timer.print()

    my_timer.delete('First')
    try:
        my_timer.read('First')
    except ArgumentError as e:
        print(e.message)
