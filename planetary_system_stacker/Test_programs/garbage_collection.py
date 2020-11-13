import gc
from math import sqrt
from time import sleep

import psutil
from numpy import full


def available_ram():
    virtual_memory = dict(psutil.virtual_memory()._asdict())
    available_ram_current = virtual_memory['available'] / 1e9
    return "{:4.2f}".format(available_ram_current)

# Look up the available RAM (without paging)

print ("Available RAM at start: " + available_ram())

object_size = 5.5
extension = int(sqrt(object_size*1e9 / 8.))
array = full((extension, extension), 1., dtype=float)
print("Available RAM after array allocation: " + available_ram())

del array

# Force the garbage collector to release unreferenced objects.
gc.collect()

for iter in range(10):
    print("Available RAM after array deletion: " + available_ram())
    sleep(0.1)