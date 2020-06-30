FROM continuumio/anaconda3:2020.02

WORKDIR /analysis

RUN conda install -c conda-forge fbprophet
RUN pip install sktime
