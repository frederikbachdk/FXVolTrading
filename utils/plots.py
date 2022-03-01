import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


def plot_returns(df:pd.DataFrame):
    plt.subplots(figsize=(12, 9))
    plt.plot(df["log_ret"], linestyle="-", color="b")
    plt.ylabel("Returns (%)")

    plt.gcf().autofmt_xdate()
    myFmt = mdates.DateFormatter("%b-%Y")
    plt.gca().set_xlim(df['log_ret'].index.min(), df['log_ret'].index.max())
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.gca().xaxis.set_major_formatter(myFmt)
    plt.title(f"Daily returns")
    plt.show()

def plot_iv(df:pd.DataFrame):

    pass