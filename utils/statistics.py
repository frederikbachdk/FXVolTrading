import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf 
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.diagnostic import het_breuschpagan
from sympy import N

def forecasting_accuracy(df:pd.DataFrame) -> None:
    print('Out of sample forecast accuracy measures:')
    # Root Mean Squared Error
    rmse = np.sqrt(np.mean((df['forward_rolling_21d_realized_stdev'] - df['cond_vol_forecast'])**2))
    print(f"RMSE: {rmse:.2f}")
    # Mean Absolute Error
    mae = np.mean(np.abs(df['forward_rolling_21d_realized_stdev'] - df['cond_vol_forecast']))
    print(f"MAE: {mae:.2f}")
    # Theil's U
    theilu = rmse / (np.sqrt(np.mean(df['forward_rolling_21d_realized_stdev']**2)) + np.sqrt(np.mean(df['cond_vol_forecast']**2)))
    print(f"Theil U: {theilu:.2f}")
    # CDC
    indicator = np.where(
        (df['forward_rolling_21d_realized_stdev']-df['forward_rolling_21d_realized_stdev'].shift(1))*(df['cond_vol_forecast']-df['forward_rolling_21d_realized_stdev'].shift(1))>0,
        1,
        0)
    cdc = np.mean(indicator) * 100
    print(f"CDC: {cdc:.2f}")


def breusch_pagan(data):
    '''
    returns p-value of LM-test (H0 is homoscedasticity)
    '''
    ols_res = OLS(data, np.ones(len(data))).fit()
    # convert into 2d array because function wants it
    s = []
    for i in data:
        a = [1,i]
        s.append(a)
    return het_breuschpagan(ols_res.resid, np.array(s))[1]


def get_desctiptive_stats(df:pd.DataFrame, plots:bool=False):

    data = df['log_ret'].dropna()

    print('################')
    print(' Normality test ')
    print('################')

    k2, p = stats.normaltest(data,nan_policy='omit')
    alpha = 0.05

    #print("k = {:.18f}".format(k2))
    print("p = {:.5f}".format(p))

    if p < alpha:  # null hypothesis: x comes from a normal distribution
        print("The null of normality can be rejected")
        #if k2 > x: print("Distribution is fat-tailed") 
    else:
        print("The null of normality cannot be rejected")

    if plots:

        fig = plt.figure(figsize=(12, 8))
        ax1 = fig.add_subplot(1, 1, 1)
        data.hist(bins=50, ax=ax1, density=True)
        ax1.set_xlabel('Return')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Return distribution')
        xmin, xmax = plt.xlim()
        mu, std = stats.norm.fit(data) 
        x = np.linspace(xmin, xmax, 1000)
        p = stats.norm.pdf(x, mu, std)
        
        plt.plot(x, p, 'k', linewidth=2)
        plt.show()

    print('#########################')
    print(' Are returns fat-tailed? ')
    print('#########################')

    k, p = stats.kurtosistest(data, alternative='greater')  # alternativ hypotese er at dist har fat tails

    print("p = {:.5f}".format(p))

    if p < alpha: 
        print("The null of no-excess kurtosis can be rejected and distribution is fat-tailed")
    else:
        print("The null of no-excess kurtosis cannot be rejected")

    
    print('#############')
    print(' Jarque-Bera ')
    print('#############')

    p = stats.jarque_bera(data)[1]  # alternativ hypotese er at dist har fat tails

    print("p = {:.5f}".format(p))

    if p < alpha: 
        print("The null of no skewness and kurtosis=3 can be rejected")
    else:
        print("The null of no skewness and kurtosis=3 cannot be rejected")

    

    print('#################################################')
    print(' Are returns statistically signifcant from zero? ')
    print('#################################################')

    stat, p = stats.ttest_1samp(data,popmean=0, nan_policy='omit')
    print('t=%.3f, p=%.5f' % (stat, p))

    if p < alpha:  
        print("The null of zero-returns can be rejected")
    else:
        print("The null of zero-returns cannot be rejected")

    print('#################')
    print(' Autocorrelation ')
    print('#################')

    nlags = 5 

    ljung_box = acorr_ljungbox(data,lags = nlags,return_df=True)

    for i in range(nlags):
        print(f"{i+1} lag(s): p-value = {ljung_box.loc[i+1, 'lb_pvalue']}")

    if plots:
        plt.rc("figure", figsize=(12,8))
        plot_acf(data)
        plt.show()

    print('##############')
    print(' Stationarity ')
    print('##############')

    result = adfuller(data)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])

    if result[1] < alpha:  
        print("The null of non-stationarity can be rejected")
    else:
        print("The null of non-stationarity cannot be rejected")

    #for key, value in result[4].items():
    #    print('\t%s: %.3f' % (key, value))

    print('##################')
    print(' Homoscedasticity ')
    print('##################')

    p = breusch_pagan(data)
    
    print("p = {:.5f}".format(p))

    if p < alpha:  
        print("The null of homoscedasticity can be rejected")
    else:
        print("The null of homoscedasticity cannot be rejected")

    # Dunis: "The fact that our currency returns have zero unconditional mean 
    # enables us to use squared returns as a measure of their variance 
    # and absolute returns as a measure of their standard deviation or volatility"

