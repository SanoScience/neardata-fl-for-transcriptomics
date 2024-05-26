# syntax=docker/dockerfile:1

FROM python:3.10-slim-buster


WORKDIR /app

COPY ./requirements.in requirements.in
COPY ./requirements.txt requirements.txt

COPY ./setup.py setup.py

RUN pip3 install --upgrade pip
RUN pip3 install pip-tools
# RUN pip-compile requirements.in
RUN pip3 install -r requirements.txt
RUN pip3 install -e .

COPY . .

CMD python3 fl/genotypes/scripts/run_server.py --server-ip=$SERVER_IP --num-clients=$NUM_CLIENTS --num-rounds=$NUM_ROUNDS --num-local-epochs=$NUM_LOCAL_EPOCHS
