import numpy as np
import pandas as pd
import os, sys, gc, random
from typing import List, NoReturn, Union, Tuple, Optional, Text, Generic, Callable, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, QuantileTransformer
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, mean_squared_error, mean_absolute_error, f1_score
from tqdm import tqdm

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt, SARIMAX
from statsmodels.tsa.forecasting.theta import ThetaModel

# https://www.statsmodels.org/dev/examples/notebooks/generated/statespace_forecasting.html

class TS_Models(object):
    """
    Model Fitting and Prediction Class:
    
    :INPUTS:
    
    :timeseries: fbprophet input
    :model: one of 'SimpleExpSmoothing', 'Holt', 'Theta', 'SARIMAX'
    :pred_points: 28 

    :EXAMPLE:
    pred = TS_Models(ts, model='Theta', pred_points=28)

    """

    def __init__(self, timeseries : pd.DataFrame, model : str='Holt', pred_points : int=28, estimate : str='accuracy'):
        # class initializing setups
        self.timeseries = timeseries
        self.model = model
        self.pred_points = pred_points
        self.estimate = estimate
        self.quantiles = [0.005, 0.025, 0.165, 0.250, 0.500, 0.750, 0.835, 0.975, 0.995]
        self.prediction = self.run()

    def run(self):
        if self.model == 'SimpleExpSmoothing':
            fitted = SimpleExpSmoothing(self.timeseries['y'].values).fit()
            if self.estimate == 'accuracy':
                pred = fitted.forecast(self.pred_points)
            elif self.estimate == 'uncertainty':
                pred = fitted.get_forecast(steps=self.pred_points)
                pred.summary_frame(alpha=0.10)

        elif self.model == 'Holt':
            fitted = Holt(self.timeseries['y'].values).fit()
            pred = fitted.forecast(self.pred_points)

        elif self.model == 'Theta':
            tss = self.timeseries.copy()
            tss.index = tss['ds']
            fitted = ThetaModel(tss.y, method="additive").fit(use_mle=True)
            pred = fitted.forecast(self.pred_points)

        elif self.model == 'SARIMAX':
            fitted = SARIMAX(self.timeseries['y'].values, order=(2, 1, 2), trend='c').fit()
            pred = fitted.forecast(self.pred_points)

        elif self.model == 'ExponentialSmoothing':
            fitted = ExponentialSmoothing(self.timeseries['y'].values + 1, seasonal_periods=21, trend='add', seasonal='add', damped=True).fit(use_boxcox=True)
            pred = fitted.forecast(self.pred_points) - 1

        return pred