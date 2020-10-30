#!/bin/bash
# sbatch --time=100:00:00 --mem-per-cpu=8000M --account=def-lwlcmyky-ab scripts/higgs_search/grid_search.sh

# Declare variables
declare -a args1=("min_precision=0.7", "min_precision=0.8", "min_precision=0.9")
declare -a args1=("min_firing=50", "min_firing=100", "min_firing=200" )
declare -a args1=("n_overlap=20", "n_overlap=40", "n_overlap=60" )
declare -a args1=("sampling_rate=0.2", "sampling_rate=0.5", "sampling_rate=0.8" )

source $(conda info --base)/etc/profile.d/conda.sh  && conda activate yala-env
export PYTHONPATH="$HOME/projects/def-lwlcmyky-ab/pvanleeu/yala"
export PYTHONPATH="$HOME/projects/def-lwlcmyky-ab/pvanleeu/lib:$PYTHONPATH"

for arg1 in "${args1[@]}"
do
    python DEV/higgs/fitting/yala_search.py -p "$arg1"
done
