#!/bin/bash

# Run 50 times the yala simulation
for i in {1..50}
do
   echo "Simulation number $i" && python DEV/higgs/fitting/fit_yala.py && python DEV/higgs/analyzing/submit.py
done
