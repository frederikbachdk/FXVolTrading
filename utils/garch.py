import numpy as np
import pandas as pd
import sys
from typing import Literal
from arch import arch_model


import warnings
from sklearn.exceptions import ConvergenceWarning

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=ConvergenceWarning)

def get_rolling_vol_forecasts(return_series, 
                                model, 
                                horizon : int=21, 
                                fitting_end_date : str = "2021-01-01",
                                #type_forecast : Literal['rolling','recursive'] = 'rolling'
                                ):
    index = return_series.index

    start_loc = 0
    end_loc = np.where(index > fitting_end_date)[0].min()

    n_forecasts = 2+ np.where(index == index[-1])[0].min() - end_loc  # find number of forecasts to make

    forecasts = {}

    print(f"Number of forecasts: {n_forecasts}")

    for i in range(n_forecasts):
        sys.stdout.write(".")
        sys.stdout.flush()

        #if type_forecast == 'rolling':
        res = model.fit(first_obs=i, last_obs=i + end_loc, disp="off")
        #else:
        #    res = model.fit(last_obs=i + end_loc, disp="off")

        temp = res.forecast(horizon=horizon, reindex=False).variance
        fcast = temp.iloc[0]
        forecasts[fcast.name] = fcast

    return pd.DataFrame(forecasts).T