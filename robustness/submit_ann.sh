#!/usr/bin/env bash
#
#SBATCH --job-name=probabilistic_param_search
#SBATCH --partition=1080ti-short
#SBATCH --time=00-01:00:00
#SBATCH --mem=8000
#SBATCH --account=rkozma
#SBATCH --output=res_%j.txt
#SBATCH -e res_%j.err
#SBATCH --gres=gpu:1

occlusionloc=${1:-0}

echo $occlusionloc

python3 dqn_ann_0.05.py --occlusionloc $occlusionloc
exit