#!/usr/bin/env bash

for occlusion in {0..100..5}
do
    sbatch submit_percentage.sh $occlusion
done