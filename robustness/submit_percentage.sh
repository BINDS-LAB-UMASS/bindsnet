#!/usr/bin/env bash
#
#SBATCH --job-name=probabilistic_param_search
#SBATCH --partition=1080ti-long
#SBATCH --time=02-00:00:00
#SBATCH --mem=8000
#SBATCH --account=rkozma
#SBATCH --output=res_%j.txt
#SBATCH -e res_%j.err
#SBATCH --gres=gpu:1

occlusion=${1:-0}
runtime=${2:-500}

echo $occlusion $runtime

python3 dqn_snn_percentage_0.05.py @@occlusion $occlusion @@runtime $runtime
exit