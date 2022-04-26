import pandas as pd
import numpy as np

def gen_trading_signals(df:pd.DataFrame, thres_up:float, thres_down:float, days_holding_period:int = 21):
    counter = 0 
    df['direction'] = 0

    for index, row in df.iterrows():
        
        if counter > 0 : 
            counter -= 1
            continue
        else:
            
            if row['cond_forecast_to_implied'] > thres_up:
                df.at[index,'direction'] = 1
                counter = days_holding_period  

            if row['cond_forecast_to_implied'] < thres_down:
                df.at[index,'direction'] = -1
                counter = days_holding_period  

    conditions = [ 
        (df['direction'] == 1) , 
        (df['direction'] == -1),     
        (df['direction'] == 0), 
        ]
    flags = ['Buy straddle', 'Sell straddle', np.NaN]

    df['direction_flag'] = np.select(conditions, flags)

    df['returns'] = np.where(
        df['direction'] != 0, 
        df['direction'] * (df['rolling_21d_realized_stdev'].shift(-days_holding_period)- df['v1m'] )/df['v1m'], 
        0)  

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