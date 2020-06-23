# Docker
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