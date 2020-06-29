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
INPUT_DIR = '../input/m5-forecasting-uncertainty/'
OUTPUT_DIR = ''
DETREND = True
MODEL = 'prophet'
MODE = 'lb'
START_DATE = '2013-04-02'

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

###
# run
###
id_individuals = df_sale.loc[np.var(df_sale.values[:, -128:], axis=1) > 160, 'id'].values.tolist() 
# id_indviduals = ['FOODS_3_090_WI_3_validation','FOODS_3_785_CA_1_validation', 'FOODS_3_362_CA_3_validation']
def CreateTimeSeries_ind(id):
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

def run_prophet_ind(id):
    # create timeseries for fbprophet
    ts = CreateTimeSeries_ind(id)

    qs = [0.750, 0.835, 0.975, 0.995]
    pred = np.zeros((28, 9))

    # define models
    for i, q in enumerate(qs):
        # (https://towardsdatascience.com/implementing-facebook-prophet-efficiently-c241305405a3)
    #     model= Prophet(holidays=holidays, seasonality_mode='multiplicative', 
    #                    daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False,
    #                    ).add_seasonality(
    #                        name='monthly', period=30.5, fourier_order=12
    #                    ).add_seasonality(
    #                        name='daily', period=1, fourier_order=15
    #                    ).add_seasonality(
    #                        name='weekly', period=7, fourier_order=20
    #                    ).add_seasonality(
    #                        name='yearly', period=365.25, fourier_order=20
    #                    ).add_seasonality(
    #                        name='quarterly', period=365.25/4, fourier_order=5, prior_scale=8
    #                    ).add_country_holidays(country_name='US')
        model = Prophet(interval_width=q).add_country_holidays(country_name='US')
            
        # fit
        model.fit(ts)

        # predict
        forecast = model.make_future_dataframe(periods=28, include_history=False)
        forecast = model.predict(forecast)

        pred[:, i] = forecast['yhat_lower'].values.transpose()
        pred[:, -i-1] = forecast['yhat_upper'].values.transpose()
        pred[:, 4] += forecast['yhat'].values.transpose() / len(qs)
    return np.append(np.array([id]), pred.ravel())

###
# run
###

# run
print("Total IDs: {}".format(len(id_individuals)))
print(f'Parallelism on {cpu_count()} CPU')
with Pool(cpu_count()) as p:
    predictions_ind = list(p.map(run_prophet_ind, id_individuals))

# only dept_id, store_id based
df_sale_group_item = df_sale[np.hstack([['dept_id','store_id'],date_columns])].groupby(['dept_id','store_id']).sum()
df_sale_group_item = df_sale_group_item.reset_index()

# create list param
ids = []
for i in range(0,df_sale_group_item.shape[0]):
    ids = ids + [(df_sale_group_item[i:i+1]['dept_id'].values[0],df_sale_group_item[i:i+1]['store_id'].values[0])]

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

    qs = [0.750, 0.835, 0.975, 0.995]
    pred = np.zeros((28, 9))

    # define models
    for i, q in enumerate(qs):
        # (https://towardsdatascience.com/implementing-facebook-prophet-efficiently-c241305405a3)
        model= Prophet(holidays=holidays, seasonality_mode='additive', holidays_prior_scale=0.1
                    ).add_country_holidays(country_name='US')
            
        # fit
        model.fit(ts)

        # predict
        forecast = model.make_future_dataframe(periods=28, include_history=False)
        forecast = model.predict(forecast)

        pred[:, i] = forecast['yhat_lower'].values.transpose()
        pred[:, -i-1] = forecast['yhat_upper'].values.transpose()
        pred[:, 4] += forecast['yhat'].values.transpose() / len(qs)
    return np.append(np.array([dept_id,store_id]), pred.ravel())

# run
print("Total IDs: {}".format(len(ids)))
print(f'Parallelism on {cpu_count()} CPU')
with Pool(cpu_count()) as p:
    predictions = list(p.starmap(run_prophet, ids))

###
# submit or check model performance
###

qs = np.array([0.005,0.025,0.165,0.25, 0.5, 0.75, 0.835, 0.975, 0.995])
levels = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "_all_"]
couples = [("state_id", "item_id"),  ("state_id", "dept_id"),("store_id","dept_id"),
                            ("state_id", "cat_id"),("store_id","cat_id")]
cols = [f"F{i}" for i in range(1, 29)]
def dept_store_sub_format(predictions3, idx):
    df_prophet_forecast_3 = pd.DataFrame()
    for k in range(0, len(predictions3)):
        dept_id = predictions3[k][0]
        store_id = predictions3[k][1]

        df_item = df_sale.loc[(df_sale.dept_id == dept_id) & (df_sale.store_id == store_id)][['id']]
        df_item['val'] = df_sale[(df_sale.dept_id == dept_id) & (df_sale.store_id == store_id)].iloc[:, np.r_[0,-28:0]].sum(axis = 1)
        pred = predictions3[k][2:].reshape(28, 9)
        for i in range(1,29):
            df_item[f'F{i}'] = (df_item['val'] * float(pred[i-1, idx]) / df_item['val'].sum())
        df_prophet_forecast_3 = pd.concat([df_prophet_forecast_3, df_item])

    df_prophet_forecast_3 = df_prophet_forecast_3.drop('val', axis=1)
    return df_prophet_forecast_3

def sub_format(predictions, idx):
    df_prophet_forecast = pd.DataFrame()
    df_prophet_forecast['id'] = [predictions[i][0] for i in range(len(predictions))]
    for i in range(1, 29):
        df_prophet_forecast[f'F{i}'] = 0
    for k in range(0, len(predictions)):
        pred = predictions[k][1:].reshape(28, 9)
        for i in range(1, 29):
            df_prophet_forecast.loc[df_prophet_forecast['id'] == predictions[k][0], f'F{i}'] = float(pred[i-1, idx])
    return df_prophet_forecast

unc_dfs_ind = {}
for idx, q in enumerate(qs):
    tmp = sub_format(predictions_ind, idx)
    tmp = tmp.merge(df_sale[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]], how="left", on = "id")
    tmp['_all_'] = 'Total'
    unc_dfs_ind[q] = tmp

unc_dfs = {}
for idx, q in enumerate(qs):
    tmp = dept_store_sub_format(predictions, idx)
    tmp = tmp.merge(df_sale[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]], how="left", on = "id")
    tmp['_all_'] = 'Total'
    unc_dfs[q] = tmp

def get_group_preds(unc_dfs, level):
    df = pd.DataFrame()
    for q in qs:
        tmp = unc_dfs[q].groupby(level)[cols].agg('sum').reset_index()
        df = pd.concat([df, tmp])
    q = np.repeat(qs, len(tmp))
    if level != "id":
        df["id"] = [f"{lev}_X_{q:.3f}_validation" for lev, q in zip(df[level].values, q)]
    else:
        df["id"] = [f"{lev.replace('_validation', '')}_{q:.3f}_validation" for lev, q in zip(df[level].values, q)]
    df = df[["id"]+list(cols)]
    return df

def get_couple_group_preds(unc_dfs, level1, level2):
    df = pd.DataFrame()
    for q in qs:
        tmp = unc_dfs[q].groupby([level1, level2])[cols].agg('sum').reset_index()
        df = pd.concat([df, tmp])
    q = np.repeat(qs, len(tmp))
    df["id"] = [f"{lev1}_{lev2}_{q:.3f}_validation" for lev1,lev2, q in 
                zip(df[level1].values,df[level2].values, q)]
    df = df[["id"]+list(cols)]
    return df

def make_submission(sub, level, couples):
    df = []
    for level in levels :
        df.append(get_group_preds(sub, level))
    for level1,level2 in couples:
        df.append(get_couple_group_preds(sub, level1, level2))
    df = pd.concat(df, axis=0, sort=False)
    df.reset_index(drop=True, inplace=True)
    df = pd.concat([df, df] , axis=0, sort=False)
    df.reset_index(drop=True, inplace=True)
    df.loc[df.index >= len(df.index)//2, "id"] = df.loc[df.index >= len(df.index)//2, "id"].str.replace(
                                        "_validation$", "_evaluation")
    return df

# make a submit file
df_sub_ind = make_submission(unc_dfs_ind, levels, couples)
df_sub = make_submission(unc_dfs, levels, couples)

for i in df_sub_ind['id'].values.tolist():
    df_sub.loc[df_sub['id'] == i, cols] = df_sub_ind.loc[df_sub_ind['id'] == i, cols]

df_sub.to_csv(OUTPUT_DIR + f'submission_{MODEL}.csv', index=False)
print('Submission file saved!')

print(df_sub.shape)
df_sub.head()