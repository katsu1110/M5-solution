# Summary - Ensemble of Statistical Forecasting Models
NOTE THAT THIS APPROACH DID NOT YIELD ANY GOOD SCORES:D!

In the M5 competitions I was in a team called *YJ kagglers* and responsible for forecasting with statistical models.

In the previous M4 competition, [weighted ensemble of statistical models](https://www.sciencedirect.com/science/article/pii/S0169207019301190) won the 3rd place. It sounded to me that 1) a time-series forecasting is still very hard for even a sophisticated machine learning model, and 2) simple statistical models may work better than GBDT or NN models.

## What I did

My approach was therefore an ensemble of statistical forecasting models enpowered by [sktime](https://github.com/alan-turing-institute/sktime), [prophet](https://facebook.github.io/prophet/), and [statsmodels](https://github.com/statsmodels/statsmodels).

The following models are used:

- Prophet
- ARIMA model
- Theta method model
- Exponential Smoothing model
- Naive model with 'seasonal_last' strategy
- Naive model with 'last' strategy


These models took the univariate (demand) input and forecast the subsequent 28 days demands on a dept_id and store_id basis, and then divided the predicted sum into the prediction of each item based on the original ratio to the sum. Only for items with high-variance, models were fitted individually. 

## What I did not do

- Running models on individual IDs (> 30,000). It just took very long to complete, but might be better given the diversity of time-series of items' demands.
- Uncertainty competition: tried it but confidence intervals yielded by statistical models weren't great in terms of the score. 

# Docker
This code (run_accuracy.py) uses the paralell processing, so running it locally may fail.

To reproduce the results, using docker is recommended.

How to execute with docker is the following. Note that docker / docker-compose are already in your local environment.

## Configure environment variables

```
cp project.env .env #  step1: copy environment variables to .env
nano .env #  step2: edit, if necessary
```

In ```.env```...

INPUT_DIR: input directory
OUTPUT_DIR: output directory

Executing a script is done within the docker container.

## Build

```
docker-compose up -d --build  # step3: build the image to initiate container
```

## Run

```
docker exec -it katsu-m5-conda bash
(base) root@0f8f3ab4a6a1:/analysis# cd script
(base) root@0f8f3ab4a6a1:/analysis/script# python run_accuracy.py 'SktimeEnsemble' 'sub'
```