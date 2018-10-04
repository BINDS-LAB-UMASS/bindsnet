#!/usr/bin/env bash

for layer1scale in 1 2 3 4 5 6 7 8 9 10
do
    sbatch submit.sh $layer1scale
done