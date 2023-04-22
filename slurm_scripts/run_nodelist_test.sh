#!/bin/bash

#SBATCH -J flower-tutorial
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH -A plgsano4-cpu
#SBATCH -p plgrid
#SBATCH --output="output.out"

module add .plgrid
module add plgrid/tools/python/3.10.4-gcccore-11.3.0


cd ..
# requires running run_configure_venv.sh prior
source ./venv/bin/activate

SERVER_NODE=${SLURM_JOB_NODELIST:0:7}
SERVER_NODE=${SERVER_NODE//[}
echo $SERVER_NODE

srun --ntasks=1 --nodelist=$SERVER_NODE --output="./slurm_scripts/output_server.out" python fl/flower_tutorial/scripts/run_server.py --server-ip=$SERVER_NODE &
#srun --ntasks=1 --nodelist=$SERVER_NODE echo $SLURM_JOB_NODELIST &
#srun --ntasks=1 echo $SLURM_JOB_NODELIST & 
srun --ntasks=1 --output="./slurm_scripts/output_client_1.out" python fl/flower_tutorial/scripts/run_client.py --server-ip=$SERVER_NODE &
srun --ntasks=1 --output="./slurm_scripts/output_client_2.out" python fl/flower_tutorial/scripts/run_client.py --server-ip=$SERVER_NODE &
wait
