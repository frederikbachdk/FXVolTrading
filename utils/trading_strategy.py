import pandas as pd
import numpy as np

def gen_trading_signals(df:pd.DataFrame, 
                        thres_up:float, 
                        thres_down:float, 
                        days_holding_period:int = 21,
                        ):
    ''' Generates trading signals:
    If days holding period is below 21, we close the staddle by buying the opposite position and thus closing out at prevailing implied vol
    '''

    assert days_holding_period > 0, 'days_holding_period must be greater than 0'
    assert days_holding_period < 22, 'days_holding_period cannot be more than 21 business days'

    # gen signals
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

    # returns in vol points 
    df['gearing'] = np.where(
        df['direction'] != 0, 
        1 + np.abs(df['cond_vol_forecast'] - df['v1m']) / df['v1m']/thres_up,  
        0)  

    if days_holding_period == 21: # hold to maturity: 
        # for buying (dir=1) straddle, we get realized vol over period less implied vol at inception adj. for tx cost
        df['returns'] = df['direction'] * (df['forward_rolling_21d_realized_stdev']- df['v1m']) - 2 * df['normalized_bid_ask_spread']
        df['returns_w_gearing'] = df['gearing']* df['direction'] * (df['forward_rolling_21d_realized_stdev']- df['v1m']) - 2 * df['normalized_bid_ask_spread']
    else: # sell option before maturity: 
        # for buying (dir=1) straddle, we get implied vol at time of selling (t+holding_days) less implied vol at inception adj. for tx cost
        df['v1m_close_trade'] = df['v1m'].shift(-days_holding_period)  # using .shift(-days) means we take the observation days out in the future (when time is ascending)
        df['returns'] = df['direction'] * (df['v1m_close_trade']- df['v1m']) - 2 * df['normalized_bid_ask_spread']
        df['returns_w_gearing'] = df['gearing']* df['direction'] * (df['v1m_close_trade']- df['v1m']) - 2 * df['normalized_bid_ask_spread']
    
    # plot formating
    conditions = [ 
        (df['direction'] == 1) , 
        (df['direction'] == -1),     
        (df['direction'] == 0), 
        ]
    flags = ['Bought straddle', 'Sold straddle', np.NaN]

    df['direction_flag'] = np.select(conditions, flags)


def calc_pnl(df:pd.DataFrame, plot:bool=False, return_df:bool=True) -> pd.DataFrame:
    ''' Needs to be called AFTER we have generated trading signals
    '''
    if 'v1m_close_trade' in df.columns:
        columns = [
            'direction',
            'v1m',
            'v1m_close_trade',
            'forward_rolling_21d_realized_stdev',
            'normalized_bid_ask_spread',
            'gearing',
            'returns', 
            'returns_w_gearing'
            ]
    else:
        columns = [
            'direction',
            'v1m',
            'forward_rolling_21d_realized_stdev',
            'normalized_bid_ask_spread',
            'gearing',
            'returns', 
            'returns_w_gearing'
            ]    
    
    df_performance = df.loc[df['direction']!=0][columns].copy(deep=True)
    df_performance.loc[pd.Timestamp('2020-12-31 00:00:00')] = 0   # insert a row with primo 0% return
    df_performance.sort_index(inplace=True)
    df_performance['normalized_pnl'] = np.cumprod(1 + df_performance['returns'].values/100) - 1
    df_performance['normalized_pnl_w_gearing'] = np.cumprod(1 + df_performance['returns_w_gearing'].values/100) - 1

    if plot:
        (df_performance[['normalized_pnl', 'normalized_pnl_w_gearing']]*100).plot(#legend=['Normalized PnL', 'Normalized PnL w. Gearing'], 
                                                                    xlabel='Date', 
                                                                    xlim=[df_performance.index.min(), df_performance.index.max()],
                                                                    ylabel='Normalized PnL (%)');
    print('Number of trades: ', len(df_performance['direction'])- 1)                                                         
    print('Mean return {:.2f}%'.format(df_performance['returns'].mean()))
    print('Mean bid-ask spread {:.2f}%'.format(df_performance['normalized_bid_ask_spread'].mean()))
    print('Normalized PnL: {:.2f}%'.format(100*df_performance['normalized_pnl'].loc[df_performance['normalized_pnl'].last_valid_index()]))
    print('Normalized PnL w. Gearing: {:.2f}%'.format(100*df_performance['normalized_pnl_w_gearing'].loc[df_performance['normalized_pnl_w_gearing'].last_valid_index()]))

    if return_df: return df_performance