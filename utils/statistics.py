import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf 


def get_desctiptive_stats(df:pd.DataFrame, plots:bool=False):

    print('########################')
    print('     Normality test     ')
    print('########################')

    k2, p = stats.normaltest(df['log_ret'],nan_policy='omit')
    alpha = 0.05

    #print("k = {:.18f}".format(k2))
    print("p = {:.5f}".format(p))

    if p < alpha:  # null hypothesis: x comes from a normal distribution
        print("The null of normality can be rejected")
        # if k2 > x: print("Distribution is fat-tailed") 
    else:
        print("The null of normality cannot be rejected")

    if plots:

        # plot histogram of returns along with dist of normal and student-t distributions

        pass

    print('#########################################################')
    print('     Are returns statistically signifcant from zero?     ')
    print('#########################################################')

    print('Returns ')
    stat, p = stats.ttest_1samp(df['log_ret'],popmean=0, nan_policy='omit')
    print('t=%.3f, p=%.3f' % (stat, p))

    if p < alpha:  # null hypothesis: x comes from a normal distribution
        print("The null of zero-returns can be rejected")
        # if k2 > x: print("Distribution is fat-tailed") 
    else:
        print("The null of zero-returns cannot be rejected")

    print('#########################')
    print('     Autocorrelation     ')
    print('#########################')

    # test for autocorrelation

    if plots:
        plot_acf(df['log_ret'].dropna())
        plt.show()

    # test for stationarity


    # test for homoskedasticity

    # Dunis: "The fact that our currency returns have zero unconditional mean 
    # enables us to use squared returns as a measure of their variance 
    # and absolute returns as a measure of their standard deviation or volatility"

    pass

