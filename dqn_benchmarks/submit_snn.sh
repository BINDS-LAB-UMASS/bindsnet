#!/usr/bin/env bash
#
#SBATCH --job-name=dqn_pixel
#SBATCH --partition=1080ti-long
#SBATCH --time=01-10:00:00
#SBATCH --mem=8000
#SBATCH --account=rkozma
#SBATCH --output=res_%j.txt
#SBATCH -e res_%j.err
#SBATCH --gres=gpu:1

python3 dqn_snn_0.05.py
exit