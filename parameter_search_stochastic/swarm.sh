#!/usr/bin/env bash
filename="particle_pos.txt"
while IFS='' read -r weight1 weight2 beta; do
    sbatch submit.sh $weight1 $weight2 $beta
done < "$filename"