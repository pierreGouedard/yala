#!/bin/bash

# Declare variables
declare -a samplingrates=("sampling_rate=0.1" "sampling_rate=0.4" "sampling_rate=0.8")
declare -a minfiring=("min_firing=100" "min_firing=1000" "min_firing=5000")
declare -a min_prec=("min_precision=0.6" "min_precision=0.75" "min_precision=0.85")

for sr in "${samplingrates[@]}"
do
  for mf in "${minfiring[@]}"
  do
    for mp in "${min_prec[@]}"
    do
      python DEV/higgs/fitting/yala_search.py -p "$sr" "$mf" "$mp"
    done
  done
done
