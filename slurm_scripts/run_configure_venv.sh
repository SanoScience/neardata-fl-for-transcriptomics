#!/bin/bash

#SBATCH -J flower-tutorial
#SBATCH -N 1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH -A plgsano4-cpu
#SBATCH -p plgrid
#SBATCH --output="output.out"
module add .plgrid
module add plgrid/tools/python/3.10.4-gcccore-11.3.0

cd ..
VENV_DIR="./venv"
[ ! -d $VENV_DIR ] && python3 -m venv venv
source ./venv/bin/activate
pip3 install --upgrade pip
pip3 install pip-tools
pip-compile --upgrade
pip3 install -r requirements.txt
pip3 install -e .