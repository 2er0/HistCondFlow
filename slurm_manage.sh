#!/bin/zsh

# load previous state
if [[ -f "./run.state" ]]; then
  source './run.state'
else
  lastRunNumber=0
fi

# debug mode to print commands instead of running them
debug=false
# experiment name
experiment_name="TFselfopt"
# options for dataset
# "fsb", "srb", "real"
dataset="fsb"
# version of the code
# version 1 is without numerical stability improvements
# version 2 is with simple numerical stability improvements
# version 3 is with improved numerical stability improvements
# version 4 is with improved numerical stability improvements, ST-Net and self-optimization
code_version="4"
self_optimization="True"
runNumber=0
seek_start=0
# define configurations to run as tuple: "model_type,past_range,slurm_type,run_splitting"
for x in "RealNVP,1,GPUQ,10" \
         "tcNF-base,101,GPUQ,10" \
         "tcNF-cnn,101,GPUQ,10" \
         "tcNF-mlp,101,GPUQ,10" \
         "tcNF-stateless,101,GPUQ,2" \
         "tcNF-stateful,1,GPUQ,2"
do
  # parse configuration
  IFS="," read model past node step <<< "${x}"

  # if "srb" then run each experiment as individual job
  if [[ $dataset == 'srb' ]]; then
    step=1
    max=10
  elif [[ $dataset == 'real' ]]; then
    step=1
    max=60
  elif [[ $dataset == 'statnett' || $dataset == 'aneo_complex' ]]; then
    step=1
    # increase this if more versions are added
    max=
  else
    # default to fsb dataset
    max=180
  fi

  # iterate over all the options for new jobs with start points/chunks
  for start in $(seq $seek_start $step $max)
  do
    # update run number and keep track of last run number
    runNumber=$((runNumber + 1))
    if [[ ${runNumber} -le ${lastRunNumber} ]]; then
      echo "Skipping $model $past $start | current iteration $runNumber | continued after $lastRunNumber"
      continue
    fi

    # check if new jobs can be queued, keep it at 14
    while true; do
      count=$(squeue -o "%A" -u davidbau | wc -l)
      echo "Queued/Running slurm jobs: $count"
      if [ "$count" -lt "15" ]; then
        break
      fi
      sleep 60
    done

    # check if the run is on GPU or CPU
    end=$((start + step))
    name="TFl-$model-$past-$start-$end"
    echo "$name"
    if [[ "$node" == "GPUQ" ]]; then
      gpu="--gres=gpu:1"
    else
      gpu=""
    fi
    if [[ "$node" == "CPUQ" ]]; then
      cores="--cpus-per-task=24"
    else
      cores="--cpus-per-task=10"
    fi

    # print or run the sbatch command
    if [[ $debug == true ]]; then
      echo "sbatch --partition=$node $gpu $cores --job-name=$name job.slurm $model $past $start $end $dataset $code_version $experiment_name $self_optimization"
    else
      sbatch --partition="$node" $gpu $cores --job-name="$name" job.slurm $model $past $start $end $dataset $code_version $experiment_name $self_optimization
    fi
    # update external state file to recover run state
    echo "lastRunNumber=$runNumber" > run.state
  done
done
