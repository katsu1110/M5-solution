import numpy as np
import pandas as pd
import os, sys, gc, random, math
from typing import List, NoReturn, Union, Tuple, Optional, Text, Generic, Callable, Dict
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

def detrend_seasonality(y : np.ndarray) -> np.ndarray:
    """

    perform detrending accounting for seasonalities
    (https://www.kaggle.com/abdelhakimbenechehab/multiseasonal-univariate-model-using-sarimax)

    """

    # time points
    predic1 = len(y)

    # Applying the Fourier series to the time scale
    predic_annual_cos = list(map(lambda x: math.cos(2*math.pi*x/365), predic1))
    predic_annual_sin = list(map(lambda x: math.sin(2*math.pi*x/365), predic1))

    predic_month_cos = list(map(lambda x: math.cos(2*math.pi*x/28), predic1))
    predic_month_sin = list(map(lambda x: math.sin(2*math.pi*x/28), predic1))

    predic_week_cos = list(map(lambda x: math.cos(2*math.pi*x/7), predic1))
    predic_week_sin = list(map(lambda x: math.sin(2*math.pi*x/7), predic1))

    # assembling the regressors
    reg = pd.DataFrame(list(zip(predic1, predic_annual_cos, predic_annual_sin, predic_month_cos, predic_month_sin, predic_week_cos, predic_week_sin)), 
                columns =['predic1', 'predic_annual_cos', 'predic_annual_sin', 'predic_month_cos', 'predic_month_sin', 'predic_week_cos', 'predic_week_sin']) 

    # fit a linear model
    model = LinearRegression().fit(reg, y)

    # the estimated parameters
    r2 = model.score(reg, y)
    print('coefficient of determination:', r2)

    # obtain parameters
    trend = model.intercept_ + model.coef_[0][0]*np.array(predic1)
    seas_annual = model.coef_[0][1]*np.array(predic_annual_cos) + model.coef_[0][2]*np.array(predic_annual_sin)
    seas_month = model.coef_[0][3]*np.array(predic_month_cos) + model.coef_[0][4]*np.array(predic_month_sin)
    seas_week = model.coef_[0][5]*np.array(predic_week_cos) + model.coef_[0][6]*np.array(predic_week_sin)

    trend_seas = trend + seas_annual + seas_month + seas_week

    return trend_seas

