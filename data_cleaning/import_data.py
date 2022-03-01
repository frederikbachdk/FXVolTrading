import numpy as np
import pandas as pd

def import_data(fx_pair:str) -> pd.DataFrame:
    """
    Set fx to e.g. USDRUB, EURUSD or any of the other FX pairs
    """

    df = pd.read_csv(f"../data/{fx_pair}.csv",index_col='Dates')
    df.index = pd.to_datetime(df.index)
    df.columns = [col.replace('USDRUB','') for col in df.columns.values]
    df.columns = map(str.lower, df.columns)
    df['log_ret'] = (np.log(df.px_last) - np.log(df.px_last.shift(1)))
    df['rolling_21d_realized_stdev'] = df['log_ret'].rolling(21).apply(
        lambda x: 1/21 * np.abs(x.sum())*np.sqrt(252)
    )
    return df