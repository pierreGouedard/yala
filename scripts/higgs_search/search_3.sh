#!/bin/bash

# Declare variables
declare -a args1=("min_precision=0.8" "min_precision=0.95")

source $(conda info --base)/etc/profile.d/conda.sh  && conda activate yala-env
export PYTHONPATH="$HOME/projects/def-lwlcmyky-ab/pvanleeu/yala"
export PYTHONPATH="$HOME/projects/def-lwlcmyky-ab/pvanleeu/lib:$PYTHONPATH"

for arg1 in "${args1[@]}"
do
    python DEV/higgs/fitting/yala_search.py -p "$arg1"
done
