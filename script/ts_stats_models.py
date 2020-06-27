import numpy as np
import pandas as pd
import os, sys, gc, random
from typing import List, NoReturn, Union, Tuple, Optional, Text, Generic, Callable, Dict
from tqdm import tqdm

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt, SARIMAX, VARMAX
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
        self.quantiles = [0.005, 0.025, 0.165, 0.250, 0.500]
        self.prediction = self.run()

    def run_accuracy(self):
        """

        Point Estimate        
        
        """
        if self.model == 'SimpleExpSmoothing':
            fitted = SimpleExpSmoothing(self.timeseries['y'].values).fit()
            pred = fitted.forecast(self.pred_points)

        elif self.model == 'Holt':
            fitted = Holt(self.timeseries['y'].values).fit()
            pred = fitted.forecast(self.pred_points)
        
        elif self.model == 'ExponentialSmoothing':
            fitted = ExponentialSmoothing(self.timeseries['y'].values + 1, seasonal_periods=21, trend='add', seasonal='add', damped=True).fit(use_boxcox=True)
            pred = fitted.forecast(self.pred_points) - 1

        elif self.model == 'Theta':
            tss = self.timeseries.copy()
            tss.index = tss['ds']
            fitted = ThetaModel(tss.y, method="additive").fit(use_mle=True)
            pred = fitted.forecast(self.pred_points)

        elif self.model == 'SARIMAX':
            fitted1 = SARIMAX(self.timeseries['y'].values, order=(2, 1, 2), trend='c').fit()
            fitted2 = SARIMAX(self.timeseries['y'].values, order=(3, 1, 5), trend='c').fit()
            pred = 0.5 * fitted1.forecast(self.pred_points) + 0.5 * fitted2.forecast(self.pred_points)

        elif self.model == 'VARMAX':
            fitted = VARMAX(self.timeseries['y'].values, order=(2, 1, 2), trend='c').fit()
            fitted2 = VARMAX(self.timeseries['y'].values, order=(3, 1, 5), trend='c').fit()
            pred = 0.5 * fitted1.forecast(self.pred_points) + 0.5 * fitted2.forecast(self.pred_points)

        return pred

    def run_uncertainty(self):
        """

        Uncertainty estimate
        (only available for state-space models)

        """

        if self.model == 'SimpleExpSmoothing':
            raise NotImplementedError()

        elif self.model == 'Holt':
            raise NotImplementedError()
        
        elif self.model == 'ExponentialSmoothing':
            raise NotImplementedError()

        elif self.model == 'Theta':
            tss = self.timeseries.copy()
            tss.index = tss['ds']
            fitted = ThetaModel(tss.y, method="additive").fit(use_mle=True)

        elif self.model == 'SARIMAX':
            fitted = SARIMAX(self.timeseries['y'].values, order=(2, 1, 2), trend='c').fit()

        elif self.model == 'VARMAX':
            fitted = VARMAX(self.timeseries['y'].values, order=(2, 1, 2), trend='c').fit()

        # uncertainty estimate
        uncertainties = np.zeros((self.pred_points, len(self.quantiles)))
        for i, q in enumerate(self.quantiles):
            fcast = fitted.get_forecast(steps=self.pred_points).summary_frame(alpha=q)
            uncertainties[:, i] = fcast['mean_ci_lower']
            uncertainties[:, -i-1] = fcast['mean_ci_upper']

        return uncertainties

    def run(self):
        if self.estimate == 'accuracy':
            return self.run_accuracy()
        elif self.estimate == 'uncertainty':
            return self.run_uncertainty()