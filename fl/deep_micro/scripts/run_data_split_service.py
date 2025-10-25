from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import argparse
import torch
from random import randint
import queue
import uvicorn
import json

parser = argparse.ArgumentParser("data_split_service")
parser.add_argument(
    "--service-ip",
    dest="service_ip",
    help="ip of the data split service",
    required=True,
    type=str,
)
parser.add_argument(
    "--n-splits", dest="n_splits", help="number of splits", required=True, type=int
)
parser.add_argument(
    "--n-samples",
    dest="n_samples",
    help="number of samples in the dataset",
    required=True,
    type=int,
)
parser.add_argument(
    "--manual-seed",
    dest="manual_seed",
    help="seed for random permutation",
    default=torch.manual_seed(randint(0, 1e6)),
)
args = parser.parse_args()

split_indices_queue = queue.Queue()


@asynccontextmanager
async def lifespan(app: FastAPI):
    partition_size = args.n_samples // args.n_splits
    shuffled_indices = torch.randperm(args.n_samples).tolist()
    split_indices = [
        shuffled_indices[i : i + partition_size]
        for i in range(0, args.n_samples, partition_size)
    ]
    for split in split_indices:
        split_indices_queue.put(split)
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/get_data_split")
async def get_data_split():
    if not split_indices_queue.empty():
        return {"split_indices": json.dumps(split_indices_queue.get())}
    raise HTTPException(status_code=404, detail="Item not found")


if __name__ == "__main__":
    uvicorn.run(
        "fl.deep_micro.scripts.run_data_split_service:app",
        host=args.service_ip,
        port=8080,
    )
