import numpy as np
import pandas as pd

# Definitions:
# prime number: natural number that is divisible by 1 and itself, excluding 0 and 1.
# möbius number:
# perfect power: natural number whose prime factorization consists of just 1 unique prime with arbitrary multiplicity.
# squarefree: natural number whose prime factorization consists of distinct primes.
# n-smooth number: natural number whose prime factors are all less than or equal to n.
# k-rough number natural number whose prime factors are all greater than or equal to k.

# Parameters
# ----------
N = 1000000 # Examine all natural numbers smaller than or equal to N

# Data Structures
primes = []
result = np.ones(N+1, dtype=np.int64)

for n in range(2,N+1):
    if (n%100000 == 0):
        print(n)
    prime_FLAG = True
    for p in primes:
        if (n%p == 0):
            prime_FLAG = False
            break
    result[n] = 1 if (prime_FLAG) else 1 + result[n//p]
    if (prime_FLAG):
        primes.append(n)

s_results = pd.Series(result)

# Absolute counts
# ---------------
absolute_counts = s_results.value_counts().sort_index()
print()
print('absolute counts')
print('---------------')
print(absolute_counts)

# Relative counts
# ---------------
relative_counts = absolute_counts/absolute_counts.sum()*100.0
print()
print('relative counts [%]')
print('-------------------')
print(relative_counts)