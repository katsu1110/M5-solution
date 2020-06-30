# My Approach - Ensemble of Statistical Forecasting Models
In the M5 competitions I was in a team called *YJ kagglers* and responsible for forecasting with statistical models.

My approach was therefore an ensemble of statistical forecasting models enpowered by [sktime](https://github.com/alan-turing-institute/sktime), [prophet](https://facebook.github.io/prophet/), and [statsmodels](https://github.com/statsmodels/statsmodels).

The following models are used:

- Prophet (accuracy & uncertainty)
- ARIMA model (accuracy & uncertainty)
- Theta method model (accuracy & uncertainty)
- Exponential Smoothing model (accuracy)
- Naive model with 'seasonal_last' strategy (accuracy)
- Naive model with 'mean' strategy (accuracy)


These models took the univariate (demand) input and forecast the subsequent 28 days demands on a dept_id and store_id basis. In the accuracy competition, for items with high-variance, models were fitted individually.

# Docker
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
docker-compose up --build  # step3: build the image to initiate container (in this case, jupyter notebook is launched)
```

## Jupyter notebook

Access jupyter notebook by typing ```localhost:8888``` in your favorite browser.