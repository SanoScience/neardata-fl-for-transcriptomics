# Neardata Fl For Transcriptomics
-----

**Table of Contents**

- [Environment setup](#environment-setup)
- [Running flower tutorial example](#running-flower-tutorial-example)
- [License](#license)

## Environment setup
### Compile the ```requirements.in``` file to get ```requirements.txt```
```bash
pip-compile requirements.in
```
### Create and activate your virtual environment.
```bash
python -m venv venv
source venv/bin/activate
```
### Now install packages from the requirements.txt file.
```bash
pip install -r requirements.txt
```
### Install also neardata-fl-for-transcriptomics from source to run scripts from the root dir.
```
pip install .
```
### We use NeptuneAI as the MLOps tool. The api token should be stored in .env file, so that python-dotenv package can load it as an env variable. The .env file should have the following entry, where 'xxx' is the api token:
```bash
NEPTUNE_API_TOKEN=xxx
```
## Running flower tutorial example
### Running locally
First, run the data split service, which is a http server that assigns samples to each client (example for a dataset with 60000 samples):
```bash
python3 fl/flower_tutorial/scripts/run_data_split_service.py --service-ip localhost --n-samples=60000 --n-splits 3 --manual-seed 1
```
You can instantiate the FL server by running:
```bash
python3 fl/flower_tutorial/scripts/run_server.py --num-clients 3 --server-ip localhost --num-rounds 50 --num-local-epochs 1
```
The ```--num-clients``` argument is the minimum number of clients the federated learning round can start with. The ```--server-ip``` argument is the address of the server.
The client can be instantiated by running:
```bash
python3 fl/flower_tutorial/scripts/run_client.py --server-ip localhost --data-split-service-ip localhost 
```
Clients and server are using port ```8081```.
### Running through SLURM
#### Setting up the environment
This script will create a virtual environment that can be used to run the tutorial example.
```bash
SBATCH run_configure_venv.sh
```
To run a FL workflow, you can use:
```bash
sbatch -n 4 run_flower_tutorial.sh 3
```
This script will run a parallel job a server + 3 clients on 4 nodes.
### Running through Docker Compose
You can run the workflow using Docker Compose. First, the data split service:
```bash
docker compose -f ./docker/flower_tutorial/docker-compose.yml up -d data-split-service
```
Second, instantiate the server container:
```bash
docker compose -f ./docker/flower_tutorial/docker-compose.yml up --build -d server
```
Then, you can build and start client containers:
```bash
docker compose -f ./docker/flower_tutorial/docker-compose.yml up --build -d --scale client=4
```
The ```--scale``` flag will allow you to instantiate a number of client containers, in this case, 4.
## Running genotypes use case
### Running locally
First, run the data split service, which is a http server that assigns samples to each client (example for a dataset with 60000 samples):
```bash
python3 fl/genotypes/scripts/run_data_split_service.py --service-ip localhost --n-samples=400 --n-splits 2 --manual-seed 1
```
You can instantiate the FL server by running:
```bash
python3 fl/genotypes/scripts/run_server.py --num-clients 2 --server-ip localhost --num-rounds 2 --num-local-epochs 1
```
The ```--num-clients``` argument is the minimum number of clients the federated learning round can start with. The ```--server-ip``` argument is the address of the server.
The client can be instantiated by running:
```bash
python3 fl/genotypes/scripts/run_client.py --server-ip localhost --data-split-service-ip localhost 
```
Clients and server are using port ```8081```.
### Running through Docker Compose
You can run the workflow using Docker Compose. First, the data split service:
```bash
docker compose -f ./docker/genotypes/docker-compose.yml up -d data-split-service
```
Second, instantiate the server container:
```bash
docker compose -f ./docker/genotypes/docker-compose.yml up -d server
```
Then, you can build and start client containers:
```bash
docker compose -f ./docker/genotypes/docker-compose.yml up -d --scale client=2
```
The ```--scale``` flag will allow you to instantiate a number of client containers, in this case, 4.
## License
This project is licensed under the MIT License 