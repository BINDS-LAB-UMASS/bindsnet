#!/usr/bin/env bash

for occlusion in {0..100..5}
do
    sbatch submit_percentage_ann.sh $occlusion
done