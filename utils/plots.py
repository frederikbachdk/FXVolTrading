import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Literal


def plot_returns(df:pd.DataFrame):

    df = df.dropna()

    plt.subplots(figsize=(12, 9))
    plt.plot(df["log_ret"], 
            linestyle="-", 
            #color="b"
            )
    plt.ylabel("Returns (%)")

    plt.gcf().autofmt_xdate()
    myFmt = mdates.DateFormatter("%b-%Y")
    plt.gca().set_xlim(df['log_ret'].index.min(), df['log_ret'].index.max())
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=12))
    plt.gca().xaxis.set_major_formatter(myFmt)
    
    plt.title('Daily Returns')
    
    plt.show()


def plot_iv(df:pd.DataFrame, months_out:Literal['1m','3m','1y']):

    df = df.dropna()

    plt.subplots(figsize=(12, 9))
    plt.plot(df[f"v{months_out}"], 
            linestyle="-", 
            #color="b"
            )
    plt.ylabel("IV")

    plt.gcf().autofmt_xdate()
    myFmt = mdates.DateFormatter("%b-%Y")
    plt.gca().set_xlim(df[f"v{months_out}"].index.min(), df[f"v{months_out}"].index.max())
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=12))
    plt.gca().xaxis.set_major_formatter(myFmt)
    
    plt.title('Daily Implied Volatility')
    
    plt.show()


# FIND OUT WHY SHAREX=FALSE DOESN'T WORK AND WHY DATES ARE OFF
def plot_grid(df_dict:dict, series:str = Literal['log_ret','v1m','v3m','v1y'], cols:int = 2):
    # determine number of rows, given the number of columns
    rows = math.ceil(len(df_dict.keys()) / cols)

    # create the figure with multiple axes
    fig, axes = plt.subplots(nrows=rows, 
                            ncols=cols, 
                            figsize=(12, 20), 
                            sharex=False, 
                            sharey=False
                            )

    # convert the axes from a nxn array to a (n*m)x1 array
    ax_array = axes.ravel()

    # get labels right
    if series == 'log_ret':
        label = '%'
        title = 'Daily Returns'
    else: 
        label = 'IV' 
        title = 'Daily Implied Volatility'

    # iterate through the dataframe dictionary keys and use enumerate
    for idx, key in enumerate(df_dict.keys()):
        ax = ax_array[idx]
        df = df_dict[key].dropna()
        df[series].plot(ax=ax, ylabel=label, title=key, visible=True)

        myFmt = mdates.DateFormatter("%b-%Y")
        ax.set_xlim(df[series].index.min(), df[series].index.max())
        ax.get_xaxis().set_visible(True)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=12))
        ax.xaxis.set_major_formatter(myFmt)
        ax.tick_params(axis='both', which='both', labelsize=9, labelbottom=True)

    plt.gcf().autofmt_xdate()
    #fig.suptitle(title, fontsize=16, fontweight="bold")
    #plt.tight_layout()
    plt.show()