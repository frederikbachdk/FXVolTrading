import numpy as np
import pandas as pd

def import_data(fx_pair:str) -> pd.DataFrame:
    """
    Set fx to e.g. USDRUB, EURUSD or any of the other FX pairs
    """
    # read data
    df = pd.read_csv(f"../data/{fx_pair}.csv",index_col='Dates')
    df.index = pd.to_datetime(df.index)
    df.columns = [col.replace(fx_pair,'') for col in df.columns.values]
    df.columns = map(str.lower, df.columns)
    
    # convert 
    if fx_pair in ['EURUSD', 'GBPUSD']:
        df[['px_last','px_bid', 'px_ask']] = 1 / df[['px_last','px_bid', 'px_ask']]
        # switch bid and ask!!!!
        df['px_bid'], df['px_ask'] = df['px_ask'].copy(), df['px_bid'].copy()
    
    # calc returns and rolling std dev
    df['log_ret'] = (np.log(df.px_last) - np.log(df.px_last.shift(1)))
    df['rolling_21d_realized_stdev'] = df['log_ret'].rolling(21).apply(
        lambda x: np.sqrt(252)* 1/21 * (np.sum(np.abs(x*100)))
    )

    # Calc bid-ask spread
    # Pip is an acronym for "percentage in point" or "price interest point." 
    # A pip is the smallest price move that an exchange rate can make based on forex market convention. 
    # Most currency pairs are priced out to four decimal places and the pip change is the last (fourth) decimal point. 
    # A pip is thus equivalent to 1/100 of 1% or one basis point.

    if fx_pair in ['USDJPY',]:  # For currency pairs such as the EUR/JPY and USD/JPY, the value of a pip is 1/100 divided by the exchange rate
        df['bid_ask_spread_pips'] = (df['px_ask'] - df['px_bid']) * 100 
    else:
        df['bid_ask_spread_pips'] = (df['px_ask'] - df['px_bid']) * 10000

    #df['normalized_bid_ask_spread'] = df['bid_ask_spread']/((df['px_ask'] + df['px_bid'])/2)
    
    return df