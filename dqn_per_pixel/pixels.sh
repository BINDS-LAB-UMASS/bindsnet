#!/usr/bin/env bash

for pixel in {0..6399}
do
    sbatch submit.sh $pixel
done