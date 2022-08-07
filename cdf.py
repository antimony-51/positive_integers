import numpy as np
import pandas as pd

def cdf(s, unit='rate'):
    s.rename('rv', inplace=True) # rv stands for random variable
    df = pd.DataFrame(s)
    df['proba'] = df.rank(method = 'max', pct = True)
    m_cdf = df.drop_duplicates().sort_values('rv') 
    if (unit == 'percent'):
        m_cdf *= 100.0
    return m_cdf

if __name__=='__main__':
    s = pd.Series(np.random.normal(loc = 10, scale = 0.1, size = 1000), name = 'rand_no')
    df = cdf(s)
    print(df.head())