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

# plot grid over 
def plot_grid(df_dict:dict, series:str = Literal['log_ret','v1m','v3m','v1y'], cols:int = 2):
    # determine number of rows, given the number of columns
    rows = math.ceil(len(df_dict.keys()) / cols)

    # create the figure with multiple axes
    fig, axes = plt.subplots(nrows=rows, 
                            ncols=cols, 
                            figsize=(12, 15), 
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
        ax.set_xlim(df['log_ret'].index.min(), df['log_ret'].index.max())
        ax.xaxis.label.set_visible(False)

    # last formating
    fig.set_facecolor('w')
    plt.tight_layout()
    plt.savefig(f"../figures/{series}_grid.png")
    plt.show()


def enumerate2(xs, start=0, step=2):
    for x in xs:
        yield (start, x)
        start += step

def plot_returns_and_vol(df_dict:dict, vol_period:str = Literal['1m','3m','1y']):
    # determine number of rows, given the number of columns
    ncols = 2
    nrows = math.ceil(len(df_dict.keys()) * 2 / ncols)

    # create the figure with multiple axes
    fig, axes = plt.subplots(nrows=nrows, 
                            ncols=ncols, 
                            figsize=(17, 20), # width, height in inches
                            sharex=False, 
                            sharey=False
                            )

    cols = ['Daily Returns', f"Daily {vol_period} Implied Volatility"]
    rows = list(df_dict.keys())

    # convert the axes from a nxn array to a (n*m)x1 array
    ax_array = axes.ravel()

    # iterate through the dataframe dictionary keys and use enumerate
    for idx, key in enumerate2(df_dict.keys()):
        
        ax = ax_array[idx]
        ax2 = ax_array[idx+1]
        
        df = df_dict[key].dropna()

        df['log_ret'].plot(ax=ax, ylabel='%', visible=True)
        ax.set_xlim(df['log_ret'].index.min(), df['log_ret'].index.max())
        ax.xaxis.label.set_visible(False)

        df[f"v{vol_period}"].plot(ax=ax2, ylabel='IV', visible=True)
        ax2.set_xlim(df[f"v{vol_period}"].index.min(), df[f"v{vol_period}"].index.max())
        ax2.xaxis.label.set_visible(False)

    # set row and column titles
    for ax, col in zip(axes[0], cols):
        ax.set_title(col,fontweight='bold')

    for ax, row in zip(axes[:,0], rows):
        ax.set_ylabel(row.strip('USD') + '      ', rotation=0, size='large', fontweight='bold')
    
    # last formating
    fig.set_facecolor('w')
    plt.tight_layout()
    plt.savefig(f"../figures/returns_and_v{vol_period}_grid.png")
    plt.show()
