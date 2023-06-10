# syntax=docker/dockerfile:1

FROM python:3.10-slim-buster

ENV SERVER_IP "localhost"
ENV NUM_CLIENTS 2
ENV NUM_ROUNDS 5
ENV NUM_LOCAL_EPOCHS 5


WORKDIR /app

COPY ./requirements.in requirements.in

COPY ./setup.py setup.py

RUN pip3 install --upgrade pip
RUN pip3 install pip-tools
RUN pip-compile requirements.in
RUN pip3 install -r requirements.txt
RUN pip3 install -e .

COPY . .

CMD python3 fl/flower_tutorial/scripts/run_server.py --server-ip=$SERVER_IP --num-clients=$NUM_CLIENTS --num-rounds=$NUM_ROUNDS --num-local-epochs=$NUM_LOCAL_EPOCHS
