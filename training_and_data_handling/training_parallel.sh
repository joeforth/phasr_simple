#!/bin/bash -l
# Use the current working directory
#SBATCH -D ./
# Use the current environment for this job
#SBATCH --export=ALL
# Define job name
#SBATCH -J YOLOv26_phasr
# Define a standard output file. When the job is running, %N will be replaced by the name of
# the first node where the job runs, %j will be replaced by job id number.
#SBATCH -o YOLOv26_phasr.%N.%j.out
# Define a standard error file
#SBATCH -e YOLOv26_phasr.%N.%j.err
# Request the GPU partition (gpu). We don't recommend requesting multiple partitions, as the specifications of the nodes in these partitions are different.
#SBATCH -p gpu-a-lowsmall
# Request the number of nodes
#SBATCH -N 1
# Request the number of GPUs per node to be used (if more than 1 GPU per node is required, change 1 into Ngpu, where Ngpu=2,3,4)
#SBATCH --gres=gpu:2
# Request the number of CPU cores. (There are 24 CPU cores and 4 GPUs on each GPU node in partition gpu,
# so please request 6*Ngpu CPU cores, i.e., 6 CPU cores for 1 GPU, 12 CPU cores for 2 GPUs, and so on.)
#SBATCH -n 12
# Set time limit in format a-bb:cc:dd, where a is days, b is hours, c is minutes, and d is seconds.
#SBATCH -t 1-00:00:00
# Insert your own username to get e-mail notifications (note: keep just one "#" before SBATCH)
#SBATCH --mail-user=j.forth@liverpool.ac.uk
# Notify user by email when certain event types occur
#SBATCH --mail-type=ALL

date
echo "This code is running on "
hostname
#echo "Starting running on host $HOSTNAME"
# Load the Conda environment
source ~/.bashrc           # Ensure Conda is available

conda activate phasr
cd /mnt/scratch/users/jwforth/phasr_simple

SCRIPTS=(
  train_barkla_lyaml.py
  train_barkla_mpt.py
  train_barkla_myaml.py
  train_barkla_spt.py
  train_barkla_syaml.py
)

echo "Running ${SCRIPTS[$SLURM_ARRAY_TASK_ID]}"
python3 "${SCRIPTS[$SLURM_ARRAY_TASK_ID]}"
echo "Finished - goodbye from $HOSTNAME"