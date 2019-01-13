#!/usr/bin/env bash

for occlusionloc in {0..77}
do
    for seed in {0..99..1}
    do
        sbatch submit.sh $seed $occlusionloc
    done
done