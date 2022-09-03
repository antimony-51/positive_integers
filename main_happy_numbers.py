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
from sympy import Integer, Matrix, N
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

    get_sums_of_squared_digit_counts()
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
        self.__value_counts[1] += Integer(1) # Take into account 10^n, whose sum of squared digits is 1.

    def get_sums_of_squared_digit_counts(self) -> List[Integer]:
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
        return self.__value_counts

class HappyNumberCounter():
    """Happy Number Counter

    determines the number of happy numbers smaller than or equal to 10^n based
    on the distribution of sums of squared numbers for the corresponding range.

    Attributes
    ----------
    ssdd : List of sympy Integers
        ssdd stands for sum of squared digits distribution, which is the output 
        of SumOfSquaredDigitsCalculator.get_sums_of_squared_digit_counts().

    Methods
    -------
    """

    def __init__(self, ssdd):
        self.__ssdd = ssdd
        self.__cat_happy_unhappy = None
        self.categorize_happy_unhappy()

    def categorize_happy_unhappy(self):
        self.__cat_happy_unhappy = [Integer(0)]*len(self.__ssdd)
        for i in range(1,len(self.__cat_happy_unhappy)):
            self.__cat_happy_unhappy[i] = HappyNumberCounter.is_happy(i)

    def get_happy_number_count(self):
        return Matrix(self.__ssdd).dot(Matrix(self.__cat_happy_unhappy))

    def get_happy_number_fraction(self):
        numerator = self.get_happy_number_count()
        denumerator = sum(self.__ssdd)-Integer(1)
        return float((numerator/denumerator).evalf())
    
    def get_happy_number_percentage(self):
        return self.get_happy_number_fraction()*100.0

    @staticmethod
    def is_happy(i):
        outcome = HappyNumberCounter.sum_of_squared_digits(i) 
        if (outcome == 1):
            return Integer(1)
        elif (outcome == 4): # All unhappy numbers iterate eventually through the loop that includes 4. See reference [2].
            return Integer(0)
        else:
            return HappyNumberCounter.is_happy(outcome)

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
    N = 250
    results = []

    print()
    print(f'Happy Number Script for n in [0, {N:d}]')
    print(f'---------------------------------------')

    ssd = SumOfSquaredDigitsCalculator()
    hnc = HappyNumberCounter(ssd.get_sums_of_squared_digit_counts())
    results.append((0, np.round(hnc.get_happy_number_fraction(),4), hnc.get_happy_number_count()))
    print(0, end='\r', flush=True)
    for n in range(1, N+1):
        ssd.set_digit_count(n)
        hnc = HappyNumberCounter(ssd.get_sums_of_squared_digit_counts())
        results.append((n, np.round(hnc.get_happy_number_fraction(),4), hnc.get_happy_number_count()))
        print(f'Calculating for n = {n:d}', end='\r' if (n<N) else '\n', flush=True)
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