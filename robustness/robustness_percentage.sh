#!/usr/bin/env bash

for runtime in 125 250
do
    for occlusion in {0..100..5}
    do
        sbatch submit_percentage.sh $occlusion $runtime
    done
done