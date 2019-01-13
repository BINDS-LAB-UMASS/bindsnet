#!/usr/bin/env bash
for occlusionloc in {0..77}
do
    sbatch submit_ann.sh $occlusionloc
done
