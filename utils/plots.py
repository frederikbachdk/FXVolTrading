from cProfile import label
import math
from statistics import mode
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
    plt.tight_layout()
    plt.show()

# plot grid over fx pairs
def plot_grid(df_dict:dict, series:str = Literal['log_ret','v1m','v3m','v1y','normalized_bid_ask_spread'], cols:int = 2):
    """
    Function for plotting a grid of plots for a given dictionary of dataframes.
    'series' Args:
        'log_ret',
        'v1m',
        'v3m',
        'v1y',
        'normalized_bid_ask_spread'
    """
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
    elif series == 'normalized_bid_ask_spread':
        label = '%'
        title = 'Normalized Bid/Ask Spread'
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
        df[series].plot(ax=ax, label='Normalized Bid/Ask Spread')
        if series == 'normalized_bid_ask_spread': 
            ax.axhline(y=0.25, color='r', linestyle='--', label='Dunis and Huang (2002) transaction cost of 25bp.')
            if idx == 3:
                ax.axvline(x="2016-11-06", color='black', linestyle='--', label='President Donald Trump elected')
            if idx == 5:
                ax.axvline(x="2016-06-25", color='black', linestyle='--', label='Brexit referendum')
        ax.set_title(key, fontweight='bold')
        ax.set_xlim(df['log_ret'].index.min(), df['log_ret'].index.max())
        ax.xaxis.label.set_visible(False)
        if idx in [0,2,4]: ax.set_ylabel(label)
        if idx == 0: ax.legend()



    # last formating
    fig.set_facecolor('w')
    plt.tight_layout()
    plt.savefig(f"../figures/{series}_grid.png")
    plt.show()


# plot grid over fx pairs
def plot_grid_forecasted_vs_realized(df_dict:dict, implied:bool, cols:int = 2):
    """
    Function for plotting a grid of plots for a given dictionary of dataframes.
    """
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
    for idx, key in enumerate(df_dict.keys()):
        ax = ax_array[idx]
        df = df_dict[key].dropna()
        df['forward_rolling_21d_realized_stdev'].plot(ax=ax, label='21-days-forward Realized Volatility', visible=True)
        df['cond_vol_forecast'].plot(ax=ax, label='GARCH(1,1) 21-days-ahead Volatility Forecast', visible=True)
        if implied: df['v1m'].plot(ax=ax, label='1m Implied Volatility', visible=True)
        ax.set_title(key, fontweight='bold')
        #ax.set_xlim(df['log_ret'].index.min(), df['log_ret'].index.max())
        ax.xaxis.label.set_visible(False)
    
        if idx == 0: ax.legend()
        if idx in [0,2,4]: ax.set_ylabel('Ann. Volatility')

    # last formating
    fig.set_facecolor('w')
    plt.tight_layout()
    plt.savefig(f"../figures/realized_forecasted.png")
    plt.show()



# plot return distribution grid over fx pairs (super inefficient)
def plot_return_distribution(df_dict:dict, bins: int = 50, cols:int = 2):
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
        df['log_ret'].hist(bins=bins, ax=ax, density=True)

        # fit pdf 
        xmin, xmax = df['log_ret'].min(), df['log_ret'].max()
        x = np.linspace(xmin, xmax, 1000)
        ## norm
        mu, std = stats.norm.fit(df['log_ret']) 
        norm_dist = stats.norm.pdf(x, mu, std)
        ax.plot(x, norm_dist, 'b', linewidth=1.5, label = 'fitted normal pdf')
        ## student t
        # degrees_of_freedom, loc, scale = stats.t.fit(df['log_ret']) 
        # student_dist = stats.t.pdf(x, loc, scale, degrees_of_freedom)
        # ax.plot(x, student_dist, 'r', linewidth=1.5, label = f"fitted students-t pdf (df={degrees_of_freedom:.1f})")
        
        # formating
        if idx in [4,5]: ax.set_xlabel('Return')
        if idx == 0: ax.legend()
        if idx in [0,2,4]: ax.set_ylabel('Observations')
        ax.set_title(key, fontweight='bold')
        ax.set_xlim(-0.045,0.045)

    # last formating
    fig.set_facecolor('w')
    #plt.figtext(0.5, 0.01, f"Number of bins: {bins}", ha="left", fontsize=8, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
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
def plot_trades(df: pd.DataFrame, thres_up:float, thres_down:float, include_implied:bool=False, include_forecasted:bool=False, include_realized:bool=False):
    """
    Plot the forecast/implied ratio and the straddles traded.
    """

    # check that the df indeed has the ratio we need
    if 'cond_forecast_to_implied' not in df.columns:
        return

    sns.set(rc={"figure.figsize": (12, 9)})
    #plt.title(f"Forecast/Implied Ratio (including traded straddles)", fontweight='bold')  # {pair} 

    # ratio
    ax1 = sns.lineplot(
        data=df,
        x=df.index,
        y="cond_forecast_to_implied",
        color="black",
        label="Forecast/Implied Ratio",
        lw=2.5,
    )
    # thresholds
    ax1.axhline(thres_up, linestyle='--', color='green', linewidth=1, label='Upper threshold (buy straddle territory)')
    ax1.axhline(thres_down, linestyle='--', color='red', linewidth=1, label='Lower threshold (sell straddle territory)')

    # straddles bought and sold
    markers = {"Bought straddle": "^", "Sold straddle": "v"}
    color_dict = dict(
        {
            "Bought straddle": "green",
            "Sold straddle": "red",
        }
    )
    sns.scatterplot(
        data=df.query("direction_flag in ('Bought straddle','Sold straddle')"),  
        x=df.query("direction_flag in ('Bought straddle','Sold straddle')").index,
        y="cond_forecast_to_implied", 
        hue="direction_flag", 
        style="direction_flag", 
        markers=markers, 
        s=200, 
        palette=color_dict,
        ax = ax1
    )
    ax1.legend(title='', loc='upper left',)
    ax1.set_ylabel('Forecast/Implied Ratio')

    # implied volatility and forecasted volatility
    if include_implied or include_forecasted or include_realized:
        ax2 = ax1.twinx()
        if include_implied: 
            sns.lineplot(
                data=df,
                x=df.index,
                y="v1m",
                #color="darkorange",
                label="Implied volatility (RHS)",
                lw=1.5,
                alpha=0.5,
                ax=ax2
            )

        if include_forecasted:
            sns.lineplot(
                data=df,
                x=df.index,
                y="cond_vol_forecast",
                #color="black",
                label="Forecasted volatility (RHS)",
                alpha=0.5,
                lw=1.5,
                ax=ax2
            )
        
        if include_realized:
            sns.lineplot(
                data=df,
                x=df.index,
                y="forward_rolling_21d_realized_stdev",
                #color="black",
                label="(Forward) realized volatility (RHS)",
                alpha=0.5,
                lw=1.5,
                ax=ax2
            )

        ax2.legend(title='', loc='upper right')
        ax2.set_ylabel('Annualized Volatility')
        ax2.grid(False)


    # last finishing off
    plt.gcf().autofmt_xdate()
    plt.show()
