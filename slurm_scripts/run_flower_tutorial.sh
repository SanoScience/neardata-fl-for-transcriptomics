#!/bin/bash
# usage: sbatch -n 4 run_flower_tutorial.sh 3
# will run a server and 3 clients on 4 nodes


#SBATCH -J flower-tutorial
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

srun --ntasks=1 --nodelist=$SERVER_NODE --output="./slurm_scripts/output_server.out" fl/flower_tutorial/scripts/run_data_split_service.py --n-samples=10000 --n-splits=$1 --manual-seed=1234 &
sleep 1
srun --ntasks=1 --nodelist=$SERVER_NODE --output="./slurm_scripts/output_server.out" python fl/flower_tutorial/scripts/run_server.py --server-ip=$SERVER_NODE --data-split-service-ip=$SERVER_NODE --num-clients=$1 &
sleep 5
for ((i=0;i<$1;i++))
do
srun --ntasks=1 --output="./slurm_scripts/output_client_$i.out" python fl/flower_tutorial/scripts/run_client.py --server-ip=$SERVER_NODE --client-id=$i &
done
wait
