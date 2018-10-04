#!/usr/bin/env bash

for layer1scale in 1
do
    sbatch submit.sh $layer1scale
done