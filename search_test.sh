#!/bin/bash
# sbatch --time=100:00:00 --mem-per-cpu=8000M --account=def-lwlcmyky-ab scripts/higgs_search/search_1.sh

# Declare variables
declare -a samplingrates=("sampling_rate=0.1" "sampling_rate=0.4" "sampling_rate=0.8")
declare -a min_prec=("min_precision=0.6" "min_precision=0.75" "min_precision=0.85")

source $(conda info --base)/etc/profile.d/conda.sh  && conda activate yala-env
export PYTHONPATH="$HOME/projects/def-lwlcmyky-ab/pvanleeu/yala"
export PYTHONPATH="$HOME/projects/def-lwlcmyky-ab/pvanleeu/lib:$PYTHONPATH"

echo "$PWD"

for sr in "${samplingrates[@]}"
do
  for mp in "${min_prec[@]}"
  do
    python DEV/higgs/fitting/test.py -p "$sr" min_firing=100 "$mp"
  done
done
