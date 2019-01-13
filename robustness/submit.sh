#!/usr/bin/env bash
#
#SBATCH --job-name=probabilistic_param_search
#SBATCH --partition=1080ti-long
#SBATCH --time=01-18:00:00
#SBATCH --mem=8000
#SBATCH --account=rkozma
#SBATCH --output=res_%j.txt
#SBATCH -e res_%j.err
#SBATCH --gres=gpu:1

seed=${1:-5}
occlusionloc=${2:-0}

echo $seed $occlusionloc

python3 dqn_snn_0.05.py @@seed $seed @@occlusionloc $occlusionloc
exit