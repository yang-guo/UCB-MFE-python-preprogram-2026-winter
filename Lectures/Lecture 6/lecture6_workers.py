"""Importable worker for the notebook multiprocessing demonstration."""

import math
import time


def calculate_primes(n: int) -> tuple[int, float]:
    start_time = time.time()
    primes = []
    for i in range(2, n):
        if all(i % j != 0 for j in range(2, int(math.sqrt(i)) + 1)):
            primes.append(i)
    return len(primes), time.time() - start_time

