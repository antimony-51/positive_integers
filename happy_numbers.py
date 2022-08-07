from unicodedata import digit
import numpy as np
import pandas as pd
import itertools

# Definitions:
# a natural number is a happy number if the sum of the squares of its digits is 1 after  

# Parameters
# ----------
n = 10
r = 12
N = n**r # Examine all natural numbers smaller than or equal to N

# Number of digit permutations = n**r
# Number of digit combinations with replacement = (n+r-1)!/(r! * (n-1)!)
dcs = np.array([c for c in itertools.combinations_with_replacement(range(n), r)]) # dcs stands for digit combinations; 1 digit combination per row
sdcs = dcs**2 # sdcs stands for squared digit combinations
ssdcs = np.sum(sdcs, axis=1) # ssdcs stands for sum of squared digit combinations

print(max(ssdcs))

# the array of sums of squared digit combinations is relatively short
# for the first 1000 numbers, make an exhaustive map
# for 6 digits, the maximum sum of squared digits is 486.
# for 12 digits, the maximum sum of squared digits is 972.
# also count the number of distinct outcomes. Either 1 or loop x or loop y.






