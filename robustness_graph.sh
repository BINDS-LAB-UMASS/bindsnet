#!/usr/bin/env bash

for layer1scale in 6.451346888745726
do
    for layer2scale in 71.15543069039526
    do
        for occlusionloc in {0..77}
        do
            sbatch submit.sh $layer1scale $layer2scale $occlusionloc
        done
    done
done