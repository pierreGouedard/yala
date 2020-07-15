#!/bin/bash

# Declare variables
declare -a samplingrates=("sampling_rate=0.1" "sampling_rate=0.4" "sampling_rate=0.8")
declare -a min_prec=("min_precision=0.6" "min_precision=0.75" "min_precision=0.85")

conda activate yala-env && \
export PYTHONPATH="$HOME/yala" && \
export PYTHONPATH="$HOME/lib:$PYTHONPATH"

for sr in "${samplingrates[@]}"
do
  for mp in "${min_prec[@]}"
  do
    python DEV/higgs/fitting/yala_search.py -p "$sr" min_firing=100 "$mp"
  done
done