import numpy as np
import pandas as pd

from collections import Counter

import plotly
import plotly.graph_objects as go
import plotly.express as px

# Definitions:
# ------------
# prime number: natural number that is divisible by 1 and itself, excluding 0 and 1.
# möbius function:
# perfect power: natural number whose prime factorization consists of just 1 unique prime with arbitrary multiplicity.
# squarefree: natural number whose prime factorization consists of distinct primes.
# s-smooth number: natural number whose prime factors are all less than or equal to s.
# r-rough number natural number whose prime factors are all greater than or equal to r.

# powerful numbers are divisible by the squares of their prime divisors, or equivalently, the multiplicity of each prime in its prime decomposition is at least 2.
# perfect squares are the squares of prime numbers
# pronic numbers are the multiplications of successive positive integers
# semiprime are the mutiplications of exactly 2 not-necessarily distinct primes
# sphenic numbers are the multiplications of exactly 3 distinct primes

# Parameters
# ----------
N = 1000000 # Examine all natural numbers smaller than or equal to N

# Data Structures
primes = []
result_is_prime = np.ones(N+1, dtype=np.int64)
result_smooth = np.ones(N+1, dtype=np.int64)
result_rough = N*np.ones(N+1, dtype=np.int64)
result_decomposition = [ [] for _ in range(N+1) ]

for n in range(2,N+1):
    if (n%100000 == 0):
        print(n)
    prime_FLAG = True

    smallest_divider = n
    for p in primes:
        if (n%p == 0):
            prime_FLAG = False
            smallest_divider = p
            break
                    
    result_is_prime[n] = 1 if (prime_FLAG) else 1 + result_is_prime[n//p]
    result_smooth[n] = max(smallest_divider, result_smooth[n//smallest_divider])   
    result_rough[n] = min(smallest_divider, result_rough[n//smallest_divider])  
    result_decomposition[n].append(smallest_divider)
    if not(prime_FLAG):
        result_decomposition[n].extend(result_decomposition[n//smallest_divider])

    if (prime_FLAG):
        primes.append(n)

result_mobius = [0, 1] + [((-1)**(len(decomp)) if (Counter(decomp).most_common(1)[0][1] == 1) else 0) for decomp in result_decomposition[2:]]
result_perfect_powers = [0, 0] + [(1 if ((len(list(Counter(decomp))) == 1) and (len(list(Counter(decomp).elements())) == 1)) else 0) for decomp in result_decomposition[2:]]
result_powerful_numbers = [0, 1] + [(1 if (Counter(decomp).most_common()[-1][1] >= 2) else 0) for decomp in result_decomposition[2:]]
result_semiprime = [0, 0] + [(1 if (len(list(Counter(decomp).elements())) == 2) else 0) for decomp in result_decomposition[2:]]
result_sphenic_numbers = [0, 0] + [(1 if ((len(list(Counter(decomp).elements())) == 3) and (Counter(decomp).most_common(1)[0][1] == 1)) else 0) for decomp in result_decomposition[2:]]

result_pronic = np.zeros(N+1, dtype=np.int64)
i = 0
while (i*(i+1) <= N):
    result_pronic[i*(i+1)] = 1
    i += 1

# Create and save a dataframe with positive integer properties.
# -------------------------------------------------------------
df = pd.DataFrame({
    'prime factor count': result_is_prime,
    'smoothness': result_smooth,
    'roughness': result_rough,
    'mobius function outcome': result_mobius,
    'perfect power': result_perfect_powers,
    'powerful number': result_powerful_numbers,
    'semiprime': result_semiprime,
    'sphenic number': result_sphenic_numbers,
    'pronic number': result_pronic
})
df.insert(4, 'square-free; even number of prime factors', (df['mobius function outcome']==1).astype(np.int64))
df.insert(5, 'square-free; odd number of prime factors', (df['mobius function outcome']==-1).astype(np.int64))
df.insert(6, 'has a squared prime', (df['mobius function outcome']==0).astype(np.int64))
df.insert(7, 'is prime', (df['prime factor count']==1).astype(np.int64))

df.to_csv('results.csv', index=True)
print(df.head(10))