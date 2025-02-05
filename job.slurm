#!/bin/sh
#_SBATCH --partition=GPUQ
#SBATCH --account=share
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#_SBATCH --cpus-per-task=6
#SBATCH --mem=120GB
#_SBATCH --gres=gpu:1
#_SBATCH --constraint="gpu80g"
#_SBATCH --exclusive
#_SBATCH --job-name="TiFl-1"
#SBATCH --output=slurm/gpu.%j.out
#_SBATCH --error=slurm/gpu.%j.err
#SBATCH --mail-user=<EMAIL@EMAIL.NO>
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Using $SLURM_CPUS_ON_NODE cores"
echo "Using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"

module purge
module load Python/3.10.8-GCCcore-12.2.0

source env/bin/activate

# $1 model type: 0: vanilla, 1: passthrough, 2: stateless, 3: stateful, 4: cnn encoding
# $2 past: the number of past time steps to include in the input
# $3 generator seek: the starting point of the generator
# $4 generator stop: the stopping point of the generator
# $5 dataset: the dataset to use
# $6 code version: the version of the code
# $7 project: the project name
# $8 $self_optimization: whether to use self-optimization

NO_HUB=TRUE PATH="$PATH:/usr" python experiment_run.py \
  --project "$7-$5-$6" --user "timeflow" \
  --model_types "$1" --past_range "$2" \
  --generator_seek "$3" --generator_stop "$4" \
  --dataset "$5" --code_version "$6" \
  --self_optimization "$8" \
  --slurm_id "$SLURM_JOB_ID" \
  --wandb_mode "online"
