#!/usr/bin/env bash
filename="particle_pos.txt"
while IFS='' read -r weight1 weight2; do
    sbatch submit.sh $weight1 $weight2
done < "$filename"