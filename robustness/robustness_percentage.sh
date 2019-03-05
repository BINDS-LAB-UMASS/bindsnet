#!/usr/bin/env bash

for occlusionloc in {0..100..5}
do
    sbatch submit_percentage.sh $occlusionloc
done