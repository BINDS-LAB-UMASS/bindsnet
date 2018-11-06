#!/usr/bin/env bash
#
#SBATCH --job-name=dqn_pixel
#SBATCH --partition=m40-short
#SBATCH --time=00-01:00:00
#SBATCH --mem=4000
#SBATCH --account=rkozma
#SBATCH --output=res_%j.txt
#SBATCH -e res_%j.err
#SBATCH --gres=gpu:1

pixel=${1:-0}

echo $pixel

python3 dqn_playground_spiking.py --pixel $pixel
exit