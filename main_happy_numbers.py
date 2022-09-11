# References
# ----------
# [1] https://oeis.org/A068571
# [2] https://oeis.org/A007770
# [3] The happy number counting algorithm was proposed by Brian Wolf: https://oeis.org/wiki/User:Bryan_Wolf
# [4] https://en.wikipedia.org/wiki/Happy_number

# Libraries
# ---------
import os
from typing import List
from sympy import Integer, Matrix, N, Float
import numpy as np
import pandas as pd
import plotly.express as px
cds = ['#19609f', '#0c304f', '#72b1e9'] # discrete color sequence with 3 colors

# Classes
# -------
# SumOfSquaredDigitsCalculator determines the distribution of sums of squared 
# digits for non-negative numbers in [0, 10^n].
#
# HappyNumberCalculator determines the absolute and relative frequencies of
# happy numbers in [0, 10^n].

class SumOfSquaredDigitsCalculator():
    """Calculate the sum of squared digits for all positive integers 
    up to 10^n.

    Attributes
    ----------
    n : int
        The sum of squared digits is determined for the first 10^n 
        positive integers including 0 and 10^n

    Constructor
    -----------
    SumOfSquaredDigitsCalculator(n:int=0)
        Initialize a SumOfSquaredDigitsCalculator for n=0 by default or
        directly specify n>0.

    Methods
    -------
    set_digit_count(n:int=0)
        Calculate the sum of squared digits for all non-negative integers
        in [0, 10^n], where n >= 0.

    get_sums_of_squared_digits_distribution()
        Get a list of sum of squared digits counts
    """

    number_of_digits = 10
    squared_digits = [d**2 for d in range(10)] # A list of squared digits is frequently needed. Store it therefore in memory.
    offset = 9**2 # Wolf's algorithm checks up to 9**2 integers that are smaller than the investigated integer. An offset of 9**2 indices is therefore frequently used.

    def __init__(self, n:int=0):
        """Initialize a SumOfSquaredDigitsCalculator object.

        Parameters
        ----------
        n : int (optional; n=0 by default)
            All non-negative integers in [0, 10^n] are considered.

        Returns
        -------
        a SumOfSquaredDigitsCalculator object.
        """
        self.set_digit_count(n)
        
    def set_digit_count(self, n:int=0) -> None:
        """Set an arbitrary exponent for the upper bound.

        Parameters
        ----------
        n : int (optional; n=0 by default)
            All non-negative integers in [0, 10^n] are considered.

        Returns
        -------
        None
        """
        if (n == 0):
            self.__n_digits = n
            self.__value_counts = [Integer(1), Integer(0)]
        elif ((self.__n_digits == 0) and (n >= 1)):
            self.__n_digits = 1
            self.__value_counts = [Integer(0)]*(SumOfSquaredDigitsCalculator.offset + 1)
            for sd in SumOfSquaredDigitsCalculator.squared_digits:
                self.__value_counts[sd] = Integer(1)

        if (self.__n_digits < n):
            self.__value_counts = [Integer(0)]*SumOfSquaredDigitsCalculator.offset + self.__value_counts
            while (self.__n_digits < n):
                self.__value_counts.extend([Integer(0)]*SumOfSquaredDigitsCalculator.offset)
                for i in range(len(self.__value_counts)-1, SumOfSquaredDigitsCalculator.offset, -1):
                    self.__value_counts[i] = sum([self.__value_counts[i - sd] for sd in SumOfSquaredDigitsCalculator.squared_digits])
                self.__n_digits += 1
            self.__value_counts = self.__value_counts[SumOfSquaredDigitsCalculator.offset:]
        # self.__value_counts[1] += Integer(1) # Take into account 10^n, whose sum of squared digits is 1.

    def get_sums_of_squared_digits_distribution(self) -> List[Integer]:
        """Get the distribution of sums of squared digits.

        Parameters
        ----------
        n : int (optional; n=0 by default)
            All non-negative integers in [0, 10^n] are considered.

        Returns
        -------
        List of sympy Integers
            The indices denote the outcomes, i.e.,  the sum of squared digits,
            and the values denote the absolute frequencies of occurence for
            all non-negative integers in [0, 10^n].
        """
        # adjusted_value_counts = self.__value_counts
        # adjusted_value_counts[1] += Integer(1) # Take into account 10^n, whose sum of squared digits is 1.
        return self.__value_counts

class HappyNumberCounter():
    """Happy Number Counter

    determines the number of happy numbers smaller than or equal to 10^n based
    on the distribution of sums of squared numbers for the corresponding range.

    Attributes
    ----------
    ssdd : List of sympy Integers
        ssdd stands for sum of squared digits distribution, which is the output 
        of SumOfSquaredDigitsCalculator.get_sums_of_squared_digits_distribution().

    Methods
    -------
    """

    memo_size = 200000

    def __init__(self):
        self.__memo_size = HappyNumberCounter.memo_size
        self.__cat_happy_unhappy = [Integer(0)] + [None]*self.__memo_size 
        self.__iteration_count = [0] + [None]*self.__memo_size 
        self.categorize_happy_unhappy()

    def categorize_happy_unhappy(self):
        for i in range(len(self.__cat_happy_unhappy)-1, 0, -1):
            if (self.__cat_happy_unhappy[i] is None):
                self.is_happy(i)

    def get_happy_number_count(self, ssdc:SumOfSquaredDigitsCalculator):
        return Integer(1) + Matrix(ssdc.get_sums_of_squared_digits_distribution()).dot(Matrix(self.__cat_happy_unhappy[:len(ssdc.get_sums_of_squared_digits_distribution())]))
        

    def get_happy_number_fraction(self, ssdc:SumOfSquaredDigitsCalculator):
        numerator = self.get_happy_number_count(ssdc)
        denumerator = sum(ssdc.get_sums_of_squared_digits_distribution())#-Integer(1)
        return float((numerator/denumerator).evalf())
    
    def get_happy_number_percentage(self, ssdc:SumOfSquaredDigitsCalculator):
        return self.get_happy_number_fraction(ssdc)*100.0

    def get_happy_number_iterations(self, ssdc:SumOfSquaredDigitsCalculator):
        return [(ic, freq) for ic,freq,b in zip(self.__iteration_count,ssdc.get_sums_of_squared_digits_distribution(),self.__cat_happy_unhappy) if b == Integer(1)]

    def is_happy(self, i):
        outcome = HappyNumberCounter.sum_of_squared_digits(i) 
        if (self.__cat_happy_unhappy[outcome] is not None):
            self.__cat_happy_unhappy[i] = self.__cat_happy_unhappy[outcome]
            self.__iteration_count[i] = self.__iteration_count[outcome] + 1
        elif (outcome == 1):
            self.__cat_happy_unhappy[i] = Integer(1)
            self.__iteration_count[i] = 1
        elif (outcome == 4): # All unhappy numbers iterate eventually through the loop that includes 4. See reference [2].
            self.__cat_happy_unhappy[i] = Integer(0)
            self.__iteration_count[i] = 1
        else:
            self.__cat_happy_unhappy[i] = self.is_happy(outcome)
            self.__iteration_count[i] = self.__iteration_count[outcome] + 1
        return self.__cat_happy_unhappy[i]

    @staticmethod
    def sum_of_squared_digits(i:int) -> int:
        outcome = 0
        while (i > 0):
            outcome += (i%10)**2
            i //= 10
        return outcome

# Happy Number Script
# -------------------
# Goal 1: Determine the percentage of happy numbers in [0, 10^n].
if __name__ == '__main__':
    base_path = 'results\\results_happy_numbers'
    sequence_id = 'A068571'
    N = 101
    n_iter_view = [10**i for i in range(1,5)]
    results = []

    print()
    print(f'Happy Number Script for n in [0, {N:d}]')
    print(f'---------------------------------------')

    ssdc = SumOfSquaredDigitsCalculator()
    hnc = HappyNumberCounter()
    results.append((0, np.round(hnc.get_happy_number_fraction(ssdc),4), hnc.get_happy_number_count(ssdc)))
    print(0, end='\r', flush=True)
    for n in range(1, N+1):
        ssdc.set_digit_count(n)
        # hnc = HappyNumberCounter(ssdc.get_sums_of_squared_digits_distribution())
        # hnc.set_sum_of_squared_digits_distribution(ssdc.get_sums_of_squared_digits_distribution())
        results.append((n, np.round(hnc.get_happy_number_fraction(ssdc),4), hnc.get_happy_number_count(ssdc)))
        print(f'Calculating for n = {n:d}', end='\r' if (n<N) else '\n', flush=True)

        if n in n_iter_view:
            # print iters
            # -----------
            result = hnc.get_happy_number_iterations(ssdc)
            df = pd.DataFrame(result, columns=['iter_count', 'frequency'])
            # print(df.groupby('iter_count')['frequency'].sum().astype(float)/float(df['frequency'].sum()))

            # df processing
            total = Float(df['frequency'].sum()) + Float(1.0)
            parts = df.groupby('iter_count')['frequency'].sum()
            result = [Integer(1), Integer(n-1)]
            for i in parts.index:
                result.append(parts[i])
            result[2] -= Integer(n-1)
            result_symbolic = [r/total for r in result]
            result_numeric = [float(r) for r in result_symbolic]

            df_iter = pd.DataFrame({'iter_count': range(len(result_numeric)), 'numeric fraction': result_numeric, 'symbolic fraction': result_symbolic})
            df_iter.to_csv(os.path.join(base_path, 'iteration_distribution_for_n_' + str(n) + '.csv'), index=False)
            
            # print('symbolic')
            # print(result_symbolic)
            print(n)
            print('numeric')
            print(result_numeric)

    print('Calculations done!')
    print()

    df = pd.DataFrame(results, columns=['n', 'percentage', 'count'])
    df.to_csv(os.path.join(base_path, sequence_id + '.csv'), index=False)

    fig = px.scatter(
        df, 
        x='n', 
        y='percentage', 
        color_discrete_sequence=cds,
        title='Percentage of Happy Numbers Smaller Than or Equal to 10^n.',
        template='plotly_white'
        )
    fig.update_layout(
        yaxis=dict(tickformat=".0%")
    )
    fig.update_traces(hovertemplate='%{y} of positive integers in ]0, 10^%{x}] is happy')
    fig.show()
    fig.write_html(os.path.join(base_path, sequence_id + '.html'))

'''
    result = hnc.get_happy_number_iterations(ssdc)
    df = pd.DataFrame(result, columns=['iter_count', 'frequency'])
    # print(df.groupby('iter_count')['frequency'].sum().astype(float)/float(df['frequency'].sum()))

    # df processing
    total = Float(df['frequency'].sum())
    parts = df.groupby('iter_count')['frequency'].sum()
    result = [Integer(1), Integer(N-1)]
    for i in parts.index:
        result.append(parts[i])
    result[2] -= Integer(N)
    result_symbolic = [r/total for r in result]
    result_numeric = [float(r) for r in result_symbolic]

    print('symbolic')
    print(result_symbolic)
    print('numeric')
    print(result_numeric)
'''