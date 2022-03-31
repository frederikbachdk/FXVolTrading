import math
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Literal
from utils.helper_functions import enumerate2

# function for plotting returns
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
    plt.set_facecolor('w')
    plt.tight_layout()
    plt.show()

# function for plotting implied volatility
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
    plt.set_facecolor('w')
    plt.tight_layout()
    plt.show()

# plot grid over fx pairs
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

    # get labels right
    if series == 'log_ret':
        label = '%'
        title = 'Daily Returns'
    else: 
        label = 'IV' 
        title = 'Daily Implied Volatility'

    plt.title(title)

    # convert the axes from a nxn array to a (n*m)x1 array
    ax_array = axes.ravel()

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



# plot return distribution grid over fx pairs
def plot_return_distribution(df_dict:dict, cols:int = 2):
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

    # iterate through the dataframe dictionary keys and use enumerate
    for idx, key in enumerate(df_dict.keys(),):
        ax = ax_array[idx]
        df = df_dict[key].dropna()
        df['log_ret'].hist(bins=50, ax=ax, density=True)
        ax.set_xlabel('Return')
        # if idx in [0,2,4]: ax.set_ylabel('Y-LABEL')
        ax.set_title(key)

        # fit normal distribution
        xmin, xmax = df['log_ret'].min(), df['log_ret'].max()
        mu, std = stats.norm.fit(df['log_ret']) 
        x = np.linspace(xmin, xmax, 1000)
        p = stats.norm.pdf(x, mu, std)
        ax.plot(x, p, 'k', linewidth=2)

    # last formating
    fig.set_facecolor('w')
    plt.tight_layout()
    plt.savefig(f"../figures/return_dist.png")
    plt.show()

       

# plot pairs x {return, iv} grid
def plot_returns_and_vol(df_dict:dict, vol_period:str = Literal['1m','3m','1y']):
    # determine number of rows, given the number of columns
    ncols = 2
    nrows = math.ceil(len(df_dict.keys()) * 2 / ncols)

    # create the figure with multiple axes
    fig, axes = plt.subplots(nrows=nrows, 
                            ncols=ncols, 
                            figsize=(17, 22), # width, height in inches
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
        ax.set_title(col,size=15,fontweight='bold')

    for ax, row in zip(axes[:,0], rows):
        ax.set_ylabel(row.strip('USD') + '      ', rotation=0, size=15, fontweight='bold')
    
    # last formating
    fig.set_facecolor('w')
    plt.tight_layout()
    plt.savefig(f"../figures/returns_and_v{vol_period}_grid.png")
    plt.show()

# function for plotting quotes
def plot_trades(df: pd.DataFrame, thres_up:float, thres_down:float):
    """
    Plot the forecast/implied ratio and the straddles traded.
    """

    # check that the df indeed has the ratio we need
    if 'cond_forecast_to_implied' not in df.columns:
        return

    sns.set(rc={"figure.figsize": (12, 9)})
    plt.title(f"Forecast/Implied Ratio (including traded straddles)", fontweight='bold')  # {pair} 

    # ratio
    sns.lineplot(
        data=df,
        x=df.index,
        y="cond_forecast_to_implied",
        #color="blue",
        label="Forecast/Implied Ratio",
    )

    # thresholds
    plt.axhline(thres_up, linestyle='--', color='black', label='Thresholds')
    plt.axhline(thres_down, linestyle='--', color='black')

    # straddles bought and sold
    markers = {"Buy straddle": "^", "Sell straddle": "v"}
    color_dict = dict(
        {
            "Buy straddle": "green",
            "Sell straddle": "red",
        }
    )
    sns.scatterplot(
        data=df.query("direction_flag in ('Buy straddle','Sell straddle')"),  
        x=df.query("direction_flag in ('Buy straddle','Sell straddle')").index,
        y="cond_forecast_to_implied", 
        hue="direction_flag", 
        style="direction_flag", 
        markers=markers, 
        s=100, 
        palette=color_dict
    )
    # last finishing off
    plt.ylabel('Forecast/Implied Ratio')
    plt.legend(title='', loc='upper right')
    plt.gcf().autofmt_xdate()
    plt.show()
