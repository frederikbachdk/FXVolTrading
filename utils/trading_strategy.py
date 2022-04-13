import pandas as pd
import numpy as np

def gen_trading_signals(df:pd.DataFrame,thres_up:float, thres_down:float):
    counter = 0 
    df['direction'] = 0

    for index, row in df.iterrows():
        
        if counter > 0 : 
            counter -= 1
            continue
        else:
            
            if row['cond_forecast_to_implied'] > thres_up:
                df.at[index,'direction'] = 1
                counter = 21  # 1m

            if row['cond_forecast_to_implied'] < thres_down:
                df.at[index,'direction'] = -1
                counter = 21  # 1m

    conditions = [ 
        (df['direction'] == 1) , 
        (df['direction'] == -1),     
        (df['direction'] == 0), 
        ]
    flags = ['Buy straddle', 'Sell straddle', np.NaN]

    df['direction_flag'] = np.select(conditions, flags)