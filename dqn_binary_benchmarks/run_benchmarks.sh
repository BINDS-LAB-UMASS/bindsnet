#!/usr/bin/env bash

sbatch submit_ann.sh

for seed in {0..99..1}
do
    for num_episodes in 1
    do
        sbatch submit_snn.sh $seed $num_episodes
        sbatch submit_snn_probabilistic.sh $seed $num_episodes
    done
done
