### 
# libraries
###
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
DETREND = True
MODEL = 'prophet'
MODE = 'sub'
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

# only dept_id, store_id based
df_sale_group_item = df_sale[np.hstack([['dept_id','store_id'],date_columns])].groupby(['dept_id','store_id']).sum()
df_sale_group_item = df_sale_group_item.reset_index()

def CreateTimeSeries(dept_id, store_id):
    item_series =  df_sale_group_item[(df_sale_group_item.dept_id == dept_id) & (df_sale_group_item.store_id == store_id)]
    dates = pd.DataFrame({'ds': dates_s}, index=range(len(dates_s)))
    dates['y'] = item_series[date_columns].values[0].transpose() 
    
    start_idx = np.where(dates['ds'] == START_DATE)[0][0]
    end_idx = np.where(dates['ds'] == END_DATE)[0][0]
    dates = dates.iloc[start_idx:end_idx].reset_index(drop=True)
    return dates

def run_prophet(dept_id, store_id):
    # create timeseries for fbprophet
    ts = CreateTimeSeries(dept_id, store_id)

    # define models
    model_add_wo = Prophet(seasonality_mode='additive')
    model_mul_wo = Prophet(seasonality_mode='multiplicative')
    model_add_w = Prophet(holidays=holidays, seasonality_mode='additive')
    model_mul_w = Prophet(holidays=holidays, seasonality_mode='multiplicative')
        
    # country holidays
    model_add_wo.add_country_holidays(country_name='US')
    model_mul_wo.add_country_holidays(country_name='US')
    model_add_w.add_country_holidays(country_name='US')
    model_mul_w.add_country_holidays(country_name='US')
    
    # fit
    model_add_wo.fit(ts)
    model_mul_wo.fit(ts)
    model_add_w.fit(ts)
    model_mul_w.fit(ts)

    # predict
    forecast_add_wo = model_add_wo.make_future_dataframe(periods=28, include_history=False)
    forecast_mul_wo = model_mul_wo.make_future_dataframe(periods=28, include_history=False)
    forecast_add_w = model_add_w.make_future_dataframe(periods=28, include_history=False)
    forecast_mul_w = model_mul_w.make_future_dataframe(periods=28, include_history=False)
    
    forecast_add_wo = model_add_wo.predict(forecast_add_wo)
    forecast_mul_wo = model_mul_wo.predict(forecast_mul_wo)
    forecast_add_w = model_add_w.predict(forecast_add_w)
    forecast_mul_w = model_mul_w.predict(forecast_mul_w)

    # ensemble
    pred = 0.25 * forecast_add_wo['yhat'].values.transpose() + 0.25 * forecast_mul_wo['yhat'].values.transpose() + \
        0.25 * forecast_add_w['yhat'].values.transpose() + 0.25 * forecast_mul_w['yhat'].values.transpose()
    return np.append(np.array([dept_id,store_id]), pred)

###
# run
###

# create list param
ids = []
for i in range(0,df_sale_group_item.shape[0]):
    ids = ids + [(df_sale_group_item[i:i+1]['dept_id'].values[0],df_sale_group_item[i:i+1]['store_id'].values[0])]

# run
print("Total IDs: {}".format(len(ids)))
print(f'Parallelism on {cpu_count()} CPU')
with Pool(cpu_count()) as p:
    predictions = list(p.starmap(run_prophet, ids))

###
# submit or check model performance
###
def dept_store_sub_format(predictions3):
    df_prophet_forecast_3 = pd.DataFrame()
    for k in range(0, len(predictions3)):
        dept_id = predictions3[k][0]
        store_id = predictions3[k][1]

        df_item = df_sale.loc[(df_sale.dept_id == dept_id) & (df_sale.store_id == store_id)][['id']]
        df_item['val'] = df_sale[(df_sale.dept_id == dept_id) & (df_sale.store_id == store_id)].iloc[:, np.r_[0,-28:0]].sum(axis = 1)
        for i in range(1,29):
            df_item[f'F{i}'] = (df_item['val'] * float(predictions3[k][i+1]) / df_item['val'].sum())
        df_prophet_forecast_3 = pd.concat([df_prophet_forecast_3, df_item])

    df_prophet_forecast_3 = df_prophet_forecast_3.drop('val',axis=1)
    return df_prophet_forecast_3

def make_submission(df_prophet_forecast_3):
    df_prophet_forecast_3.columns = df_sample.columns

    df_sub_eval = df_prophet_forecast_3.copy()
    df_sub_eval['id'] = df_sub_eval['id'].str.replace("validation", "evaluation")

    df_sub = pd.concat([df_prophet_forecast_3, df_sub_eval], sort=False)
    df_sub = df_sub.sort_values('id')

    # Fix negative forecast
    num = df_sub._get_numeric_data()
    num[num < 0] = 0

    df_sub.to_csv('submission.csv', index=False)

    print(f'Submission shape: {df_sub.shape}')
    return df_sub

# make a submit file
df_prophet_forecast = dept_store_sub_format(predictions)
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