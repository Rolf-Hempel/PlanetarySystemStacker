from ctypes import CDLL, byref, c_int
from time import time

from numpy import random, matmul


def mkl_set_num_threads(cores):
    mkl_rt.mkl_set_num_threads(byref(c_int(cores)))


mkl_rt_path = "C:\Python35\Lib\site-packages\\numpy\core\mkl_rt.dll"
# mkl_rt_path = "C:\Python36\Library\\bin\mkl_rt.dll"
mkl_rt = CDLL(mkl_rt_path)

ndim = 20000
iterations = 10

for number_threads in [1, mkl_rt.mkl_get_max_threads()]:
    mkl_set_num_threads(number_threads)
    print("\nNumber of threads used by mkl: " + str(number_threads))

    start = time()
    matrix = random.rand(ndim, ndim)
    vector = random.rand(ndim)
    end = time()
    print("Time for random number generation: " + str(end - start) + " seconds.")

    start = time()
    for iter in range(iterations):
        vector = matmul(matrix, vector)
    end = time()
    print("Time for " + str(iterations) + " matrix multiplies: " + str(end - start) + " seconds.")
