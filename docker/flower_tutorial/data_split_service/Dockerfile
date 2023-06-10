# syntax=docker/dockerfile:1

FROM python:3.10-slim-buster

ENV N_SAMPLES 1000
ENV N_SPLITS 2
ENV MANUAL_SEED 1

WORKDIR /app

COPY ./requirements.in requirements.in

COPY ./setup.py setup.py

RUN pip3 install --upgrade pip
RUN pip3 install pip-tools
RUN pip-compile requirements.in
RUN pip3 install -r requirements.txt
RUN pip3 install -e .

COPY ./fl/flower_tutorial/scripts/run_data_split_service.py ./fl/flower_tutorial/scripts/run_data_split_service.py

CMD python3 fl/flower_tutorial/scripts/run_data_split_service.py --n-samples=$N_SAMPLES --n-splits=$N_SPLITS --manual-seed=$MANUAL_SEED
