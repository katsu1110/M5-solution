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
    # quantiles
    qs = [0.750, 0.835, 0.975, 0.995]
    pred = np.zeros((28, 9))

    # create timeseries for fbprophet
    ts = CreateTimeSeries(dept_id, store_id)

    for i, q in enumerate(qs):
        # define models
        model_add_wo = Prophet(seasonality_mode='additive', interval_width=q)
        model_mul_wo = Prophet(seasonality_mode='multiplicative', interval_width=q)
        model_add_w = Prophet(holidays=holidays, seasonality_mode='additive', interval_width=q)
        model_mul_w = Prophet(holidays=holidays, seasonality_mode='multiplicative', interval_width=q)
            
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
        pred[:, i] = 0.25 * forecast_add_wo['yhat_lower'].values.transpose() + 0.25 * forecast_mul_wo['yhat_lower'].values.transpose() + \
            0.25 * forecast_add_w['yhat_lower'].values.transpose() + 0.25 * forecast_mul_w['yhat_lower'].values.transpose()
        pred[:, -i-1] = 0.25 * forecast_add_wo['yhat_upper'].values.transpose() + 0.25 * forecast_mul_wo['yhat_upper'].values.transpose() + \
            0.25 * forecast_add_w['yhat_upper'].values.transpose() + 0.25 * forecast_mul_w['yhat_upper'].values.transpose()
        pred[:, 4] += 0.25 * forecast_add_wo['yhat'].values.transpose() + 0.25 * forecast_mul_wo['yhat'].values.transpose() + \
            0.25 * forecast_add_w['yhat'].values.transpose() + 0.25 * forecast_mul_w['yhat'].values.transpose()
    pred[:, 4] /= 4
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
qs = np.array([0.005,0.025,0.165,0.25, 0.5, 0.75, 0.835, 0.975, 0.995])
levels = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "_all_"]
couples = [("state_id", "item_id"),  ("state_id", "dept_id"),("store_id","dept_id"),
                            ("state_id", "cat_id"),("store_id","cat_id")]
cols = [f"F{i}" for i in range(1, 29)]
VALID = []
EVAL = []
def dept_store_sub_format(predictions3, idx):
    df_prophet_forecast_3 = pd.DataFrame()
    for k in range(0, len(predictions3)):
        dept_id = predictions3[k][0]
        store_id = predictions3[k][1]

        df_item = df_sale.loc[(df_sale.dept_id == dept_id) & (df_sale.store_id == store_id)][['id']]
        df_item['val'] = df_sale[(df_sale.dept_id == dept_id) & (df_sale.store_id == store_id)].iloc[:, np.r_[0,-28:0]].sum(axis = 1)
        for i in range(1,29):
            df_item[f'F{i}'] = (df_item['val'] * float(predictions3[k][i+1, idx]) / df_item['val'].sum())
        df_prophet_forecast_3 = pd.concat([df_prophet_forecast_3, df_item])

    df_prophet_forecast_3 = df_prophet_forecast_3.drop('val', axis=1)
    return df_prophet_forecast_3

unc_dfs = {}
for idx, q in enumerate(qs):
    tmp = dept_store_sub_format(predictions, idx)
    tmp = tmp.merge(df_sale[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]], how="left", on = "id")
    tmp['_all_'] = 'Total'
    unc_dfs[q] = tmp

def get_group_preds(unc_dfs, level):
    df = pd.DataFrame()
    for q in qs:
        tmp = unc_dfs[q].groupby(level)[cols].sum()
        df = pd.concat([df, tmp], ignore_index=True)
    q = np.repeat(qs, len(df))
    if level != "id":
        df["id"] = [f"{lev}_X_{q:.3f}_validation" for lev, q in zip(df[level].values, q)]
    else:
        df["id"] = [f"{lev.replace('_validation', '')}_{q:.3f}_validation" for lev, q in zip(df[level].values, q)]
    df = df[["id"]+list(cols)]
    return df

def get_couple_group_preds(unc_dfs, level1, level2):
    df = pd.DataFrame()
    for q in qs:
        tmp = unc_dfs[q].groupby([level1, level2])[cols].sum()
        df = pd.concat([df, tmp], ignore_index=True)
    q = np.repeat(qs, len(df))
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
df_sub = make_submission(unc_dfs, level, couples)
 
df_sub.to_csv(OUTPUT_DIR + f'submission_{MODEL}.csv')
print('Submission file saved!')
