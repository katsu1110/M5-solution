### 
# libraries
###
# !pip install git+https://github.com/statsmodels/statsmodels
# from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt, SARIMAX, VARMAX

import numpy as np
import pandas as pd
import os, sys, gc
import random
import math
from typing import List, NoReturn, Union, Tuple, Optional, Text, Generic, Callable, Dict
from tqdm import tqdm

from multiprocessing import Pool, cpu_count
from fbprophet import Prophet
from sklearn.linear_model import LinearRegression

# visualize
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
from matplotlib import pyplot
from matplotlib.ticker import ScalarFormatter
sns.set_context("talk")
style.use('seaborn-colorblind')

import warnings
warnings.filterwarnings('ignore')

###
# config
###
INPUT_DIR = '../input/m5-forecasting-accuracy/'
OUTPUT_DIR = ''
ESTIMATE = 'accuracy'
MODEL = 'ExponentialSmoothing'
MODE = 'lb'
START_DATE = '2015-04-02'

if MODE == 'cv':
    END_DATE = '2016-03-28'
elif MODE == 'lb':
    END_DATE = '2016-04-24'
elif MODE == 'sub':
    END_DATE = '2016-05-22'

###
# load data
###
def load_data():
    # df_sale = pd.read_csv(INPUT_DIR + 'sales_train_validation.csv')
    df_calendar = pd.read_csv(INPUT_DIR + 'calendar.csv')
    df_price = pd.read_csv(INPUT_DIR + 'sell_prices.csv')
    df_sale = pd.read_csv(INPUT_DIR + 'sales_train_evaluation.csv')
    df_sample = pd.read_csv(INPUT_DIR + 'sample_submission.csv')

    # evaluation -> validation
    df_sale['id'] = df_sale['id'].str.replace("evaluation", "validation")
    return df_sale, df_calendar, df_price, df_sample
df_sale, df_calendar, df_price, df_sample = load_data()
    
###
# Format events
###
def format_holidays(df_sale):
    columns = df_sale.columns
    date_columns = columns[columns.str.contains("d_")]
    dates_s = [pd.to_datetime(df_calendar.loc[df_calendar['d'] == str_date,'date'].values[0]) for str_date in date_columns]

    tmp = df_sale[date_columns].sum()
    ignore_date = df_calendar[df_calendar['d'].isin(tmp[tmp < 10000].index.values)]['date'].values

    df_ev_1 = pd.DataFrame({'holiday': 'Event 1', 'ds': df_calendar[~df_calendar['event_name_1'].isna()]['date']})
    df_ev_2 = pd.DataFrame({'holiday': 'Event 2', 'ds': df_calendar[~df_calendar['event_name_2'].isna()]['date']})
    df_ev_3 = pd.DataFrame({'holiday': 'snap_CA', 'ds': df_calendar[df_calendar['snap_CA'] == 1]['date']})
    df_ev_4 = pd.DataFrame({'holiday': 'snap_TX', 'ds': df_calendar[df_calendar['snap_TX'] == 1]['date']})
    df_ev_5 = pd.DataFrame({'holiday': 'snap_WI', 'ds': df_calendar[df_calendar['snap_WI'] == 1]['date']})
    holidays = pd.concat((df_ev_1, df_ev_2, df_ev_3, df_ev_4, df_ev_5))

    holidays['ds'] = pd.to_datetime(holidays['ds'])
    oh_holidays = pd.concat([holidays, pd.get_dummies(holidays['holiday'])], axis=1)
    oh_holidays.drop(columns=['holiday'], inplace=True)
    return holidays, oh_holidays, date_columns, dates_s, ignore_date
holidays, oh_holidays, date_columns, dates_s, ignore_date = format_holidays(df_sale)

###
# Time series 
###

def CreateTimeSeries(id):
    item_series = df_sale[df_sale['id'] == id]
    columns = df_sale.columns
    date_columns = columns[columns.str.contains("d_")]
    dates_s = [pd.to_datetime(df_calendar.loc[df_calendar['d'] == str_date,'date'].values[0]) for str_date in date_columns]
    dates = pd.DataFrame({'ds': dates_s}, index=range(len(dates_s)))
    dates['y'] = item_series[date_columns].values.transpose()
    # Remove chirstmas date
    #dates = dates[~dates['ds'].isin(ignore_date)]
    # Remove zero day
    #dates = dates[dates['y'] > 0]        
    start_idx = np.where(dates['ds'] == START_DATE)[0][0]
    end_idx = np.where(dates['ds'] == END_DATE)[0][0]
    dates = dates.iloc[start_idx:end_idx].reset_index(drop=True)
    return dates

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
        self.quantiles = [0.005, 0.025, 0.165, 0.250]
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
            fitted = ExponentialSmoothing(self.timeseries['y'].values + 1, seasonal_periods=7, damped=True).fit(use_boxcox=True)
            pred = fitted.forecast(self.pred_points) - 1

        elif self.model == 'Theta':
            tss = self.timeseries.copy()
            tss.index = tss['ds']
            fitted = ThetaModel(tss.y).fit(use_mle=True)
            pred = fitted.forecast(self.pred_points)

        elif self.model == 'SARIMAX':
            fitted1 = SARIMAX(self.timeseries['y'].values, order=(2, 1, 2), trend='c').fit()
            fitted2 = SARIMAX(self.timeseries['y'].values, order=(3, 1, 5), trend='c').fit()
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
            fitted = SARIMAX(self.timeseries['y'].values, order=(3, 1, 5), trend='c').fit()

        # uncertainty estimate
        uncertainties = np.zeros((self.pred_points, 2 * len(self.quantiles) + 1))
        for i, q in enumerate(self.quantiles):
            fcast = fitted.get_forecast(steps=self.pred_points).summary_frame(alpha=q)
            uncertainties[:, i] = fcast['mean_ci_lower']
            uncertainties[:, -i-1] = fcast['mean_ci_upper']
            uncertainties[:, 4] += fcast['mean'] / len(self.quantiles)

        return uncertainties

    def run(self):
        if self.estimate == 'accuracy':
            return self.run_accuracy()
        elif self.estimate == 'uncertainty':
            return self.run_uncertainty()

def run_ts(id):
    # create timeseries for fbprophet
    ts = CreateTimeSeries(id)

    # define models
    pred = TS_Models(timeseries=ts, model=MODEL, pred_points=28, estimate=ESTIMATE)
    return np.append(np.array([id]), pred.prediction)

###
# run
###

# create list param
ids = df_sale['id'].values

# run
print("Total IDs: {}".format(len(ids)))
print(f'Parallelism on {cpu_count()} CPU')
with Pool(cpu_count()) as p:
    predictions = list(p.map(run_ts, ids))

###
# submit or check model performance
###
def sub_format(predictions):
    df_prophet_forecast = pd.DataFrame()
    df_prophet_forecast['id'] = [predictions[i][0] for i in range(len(predictions))]
    for i in range(1, 29):
        df_prophet_forecast[f'F{i}'] = 0
    for k in range(0, len(predictions)):
        for i in range(1, 29):
            df_prophet_forecast.loc[df_prophet_forecast['id'] == predictions[k][0], f'F{i}'] = float(predictions[k][i])
    return df_prophet_forecast

def make_submission(df_prophet_forecast):
    df_prophet_forecast.columns = df_sample.columns

    df_sub_eval = df_prophet_forecast.copy()
    df_sub_eval['id'] = df_sub_eval['id'].str.replace("validation", "evaluation")

    df_sub = pd.concat([df_prophet_forecast, df_sub_eval], sort=False)
    df_sub = df_sub.sort_values('id')

    # Fix negative forecast
    num = df_sub._get_numeric_data()
    num[num < 0] = 0
    return df_sub

# make a submit file
df_prophet_forecast = sub_format(predictions)
df_sub = make_submission(df_prophet_forecast)

df_sub.to_csv(OUTPUT_DIR + f'submission_{MODEL}.csv', index=False)
print('Submission file saved!')

print(df_sub.shape)
df_sub.head()

# compute score
if MODE != 'sub':
    ## evaluation metric
    ## from https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/133834 and edited to get scores at all levels
    class WRMSSEEvaluator(object):

        def __init__(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame):
            train_y = train_df.loc[:, train_df.columns.str.startswith('d_')]
            train_target_columns = train_y.columns.tolist()
            weight_columns = train_y.iloc[:, -28:].columns.tolist()

            train_df['all_id'] = 0  # for lv1 aggregation

            id_columns = train_df.loc[:, ~train_df.columns.str.startswith('d_')].columns.tolist()
            valid_target_columns = valid_df.loc[:, valid_df.columns.str.startswith('d_')].columns.tolist()

            if not all([c in valid_df.columns for c in id_columns]):
                valid_df = pd.concat([train_df[id_columns], valid_df], axis=1, sort=False)

            self.train_df = train_df
            self.valid_df = valid_df
            self.calendar = calendar
            self.prices = prices

            self.weight_columns = weight_columns
            self.id_columns = id_columns
            self.valid_target_columns = valid_target_columns

            weight_df = self.get_weight_df()

            self.group_ids = (
                'all_id',
                'cat_id',
                'state_id',
                'dept_id',
                'store_id',
                'item_id',
                ['state_id', 'cat_id'],
                ['state_id', 'dept_id'],
                ['store_id', 'cat_id'],
                ['store_id', 'dept_id'],
                ['item_id', 'state_id'],
                ['item_id', 'store_id']
            )

            for i, group_id in enumerate(tqdm(self.group_ids)):
                train_y = train_df.groupby(group_id)[train_target_columns].sum()
                scale = []
                for _, row in train_y.iterrows():
                    series = row.values[np.argmax(row.values != 0):]
                    scale.append(((series[1:] - series[:-1]) ** 2).mean())
                setattr(self, f'lv{i + 1}_scale', np.array(scale))
                setattr(self, f'lv{i + 1}_train_df', train_y)
                setattr(self, f'lv{i + 1}_valid_df', valid_df.groupby(group_id)[valid_target_columns].sum())

                lv_weight = weight_df.groupby(group_id)[weight_columns].sum().sum(axis=1)
                setattr(self, f'lv{i + 1}_weight', lv_weight / lv_weight.sum())

        def get_weight_df(self) -> pd.DataFrame:
            day_to_week = self.calendar.set_index('d')['wm_yr_wk'].to_dict()
            weight_df = self.train_df[['item_id', 'store_id'] + self.weight_columns].set_index(['item_id', 'store_id'])
            weight_df = weight_df.stack().reset_index().rename(columns={'level_2': 'd', 0: 'value'})
            weight_df['wm_yr_wk'] = weight_df['d'].map(day_to_week)

            weight_df = weight_df.merge(self.prices, how='left', on=['item_id', 'store_id', 'wm_yr_wk'])
            weight_df['value'] = weight_df['value'] * weight_df['sell_price']
            weight_df = weight_df.set_index(['item_id', 'store_id', 'd']).unstack(level=2)['value']
            weight_df = weight_df.loc[zip(self.train_df.item_id, self.train_df.store_id), :].reset_index(drop=True)
            weight_df = pd.concat([self.train_df[self.id_columns], weight_df], axis=1, sort=False)
            return weight_df

        def rmsse(self, valid_preds: pd.DataFrame, lv: int) -> pd.Series:
            valid_y = getattr(self, f'lv{lv}_valid_df')
            score = ((valid_y - valid_preds) ** 2).mean(axis=1)
            scale = getattr(self, f'lv{lv}_scale')
            return (score / scale).map(np.sqrt)

        def score(self, valid_preds: Union[pd.DataFrame, np.ndarray]):
            assert self.valid_df[self.valid_target_columns].shape == valid_preds.shape

            if isinstance(valid_preds, np.ndarray):
                valid_preds = pd.DataFrame(valid_preds, columns=self.valid_target_columns)

            valid_preds = pd.concat([self.valid_df[self.id_columns], valid_preds], axis=1, sort=False)

            group_ids = []
            all_scores = []
            for i, group_id in enumerate(self.group_ids):
                lv_scores = self.rmsse(valid_preds.groupby(group_id)[self.valid_target_columns].sum(), i + 1)
                weight = getattr(self, f'lv{i + 1}_weight')
                lv_scores = pd.concat([weight, lv_scores], axis=1, sort=False).prod(axis=1)
                group_ids.append(group_id)
                all_scores.append(lv_scores.sum())

            return group_ids, all_scores

    # evaluator
    if MODE == 'cv':
        evaluator = WRMSSEEvaluator(df_sale.iloc[:, :-56], df_sale.iloc[:, -56:-28], df_calendar, df_price)
        d_range = range(1886, 1914)
    elif MODE == 'lb':
        evaluator = WRMSSEEvaluator(df_sale.iloc[:, :-28], df_sale.iloc[:, -28:], df_calendar, df_price)
        d_range = range(1914, 1942)

    def get_score(evaluator, preds_valid):
        groups, scores = evaluator.score(preds_valid)
        for i in range(len(groups)):
            print(f"Score for group {groups[i]}: {round(scores[i], 5)}")
        print('*'*50)
        print(f"\nPublic LB Score: {round(np.mean(scores), 5)}")
        print('*'*50)

    # check CV or LB
    df_val = df_prophet_forecast[[f'F{i+1}' for i in range(28)]].copy()
    df_val.columns = [f'd_{i}' for i in d_range]
    get_score(evaluator, df_val)