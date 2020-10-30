#!/bin/bash

n=0
for entry in "DATA/submissions/higgs/search"/*
do
  echo "$entry"
  kaggle competitions submit -c higgs-boson -f "$entry" -m "submission $n" &&
  echo "submission $n done" &&
  let "n+=1"
done
