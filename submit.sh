#!/usr/bin/env bash
#
#SBATCH --job-name=probabilistic_param_search
#SBATCH --partition=1080ti-long
#SBATCH --time=10:00
#SBATCH --mem=8000
#SBATCH --account=rkozma
#SBATCH --output=../../output/dqn_playground_spiking_%j.out

layer1scale=${1:-1}
layer2scale=${2:-1}

echo $layer1scale $layer2scale

python3 dqn_playground_spiking.py --layer1scale $layer1scale --layer2scale $layer2scale
exit