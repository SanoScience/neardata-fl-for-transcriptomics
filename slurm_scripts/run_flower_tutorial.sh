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
# requires running run_configure_venv.sh prior
source ./venv/bin/activate

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python fl/flower_tutorial/scripts/run_server.py &
srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python fl/flower_tutorial/scripts/run_client.py &
srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python fl/flower_tutorial/scripts/run_client.py &
wait