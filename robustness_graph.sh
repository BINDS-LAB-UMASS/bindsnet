#!/usr/bin/env bash

for layer1scale in 5
do
    for layer2scale in 1
    do
        for occlusionloc in {0..77}
        do
            sbatch submit.sh $layer1scale $layer2scale $occlusionloc
        done
    done
done