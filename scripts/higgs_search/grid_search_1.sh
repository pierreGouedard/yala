#!/bin/bash
# sbatch --time=100:00:00 --mem-per-cpu=8000M --account=def-lwlcmyky-ab scripts/higgs_search/grid_search_1.sh

# Declare variables
declare -a args1=("min_precision=0.5" "min_precision=0.75" "min_precision=0.9")
declare -a args2=("min_firing=100" "min_firing=250" "min_firing=500" )
declare -a args3=("n_overlap=50" "n_overlap=100" "n_overlap=200" )
declare -a args4=("sampling_rate=0.2" "sampling_rate=0.5" "sampling_rate=0.8" )

# source $(conda info --base)/etc/profile.d/conda.sh  && conda activate yala-env

for arg1 in "${args1[@]}"
do
  for arg2 in "${args2[@]}"
    do
      for arg3 in "${args3[@]}"
        do
          for arg4 in "${args4[@]}"
            do
              python DEV/higgs/fitting/grid_search.py -p "$arg1" "$arg2" "$arg3" "$arg4"
            done
        done
    done
done
