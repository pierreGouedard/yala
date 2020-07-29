#!/bin/bash
# sbatch --time=100:00:00 --mem-per-cpu=8000M --account=def-lwlcmyky-ab scripts/higgs_search/search_1.sh

# Declare variables
declare -a args1=("n_overlap=10" "n_overlap=50" "n_overlap=100")

source $(conda info --base)/etc/profile.d/conda.sh  && conda activate yala-env
export PYTHONPATH="$HOME/projects/def-lwlcmyky-ab/pvanleeu/yala"
export PYTHONPATH="$HOME/projects/def-lwlcmyky-ab/pvanleeu/lib:$PYTHONPATH"

for arg1 in "${args1[@]}"
do
    python DEV/higgs/fitting/yala_search.py -p "$arg1" min_firing=100
done
