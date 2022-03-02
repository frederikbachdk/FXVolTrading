import pandas as pd
import numpy as np
import os
raw_data_path = 'data/raw/data.xlsx'

fx =[
    'EURUSD',
    'USDJPY',
    'GBPUSD',
    'USDRUB',
    'USDZAR',
    'USDBRL'
]

# implied vol data
raw_data_iv = pd.read_excel(raw_data_path, sheet_name='impliedvols')
raw_data_iv.set_index('Dates',inplace=True)
raw_data_iv.columns = raw_data_iv.columns.str.strip(' Curncy')

# exchange rate data
raw_data_ex = pd.read_excel(raw_data_path, sheet_name='exchangerates',header=[0, 1])

    # clean column names
raw_data_ex.columns = [' '.join(col).strip() for col in raw_data_ex.columns.values]
raw_data_ex.columns = [col.replace(' Curncy ','') for col in raw_data_ex.columns.values]
raw_data_ex.columns = [col.replace('USDEUR','EURUSD') for col in raw_data_ex.columns.values]
raw_data_ex.columns = [col.replace('USDGBP','GBPUSD') for col in raw_data_ex.columns.values]

    # set and clean index
raw_data_ex.set_index(raw_data_ex.columns[raw_data_ex.columns.str.contains('Dates')][0],inplace=True)
raw_data_ex.index.rename('Dates',inplace=True)

for pair in fx:
    raw_data_iv.filter(regex=pair).join(raw_data_ex.filter(regex=pair),how='left').to_csv(f"data/{pair}.csv")