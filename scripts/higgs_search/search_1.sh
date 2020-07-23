#!/bin/bash
# sbatch --time=100:00:00 --mem-per-cpu=8000M --account=def-lwlcmyky-ab scripts/higgs_search/search_1.sh

# Declare variables
declare -a samplingrates=("sampling_rate=1.0" "sampling_rate=0.6" "sampling_rate=0.8")
declare -a min_prec=("min_precision=0.8" "min_precision=0.9")

source $(conda info --base)/etc/profile.d/conda.sh  && conda activate yala-env
export PYTHONPATH="$HOME/projects/def-lwlcmyky-ab/pvanleeu/yala"
export PYTHONPATH="$HOME/projects/def-lwlcmyky-ab/pvanleeu/lib:$PYTHONPATH"

for sr in "${samplingrates[@]}"
do
  for mp in "${min_prec[@]}"
  do
    python DEV/higgs/fitting/yala_search.py -p "$sr" min_firing=200 "$mp"
  done
done
