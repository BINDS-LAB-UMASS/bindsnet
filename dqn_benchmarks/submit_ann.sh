#!/usr/bin/env bash
#
#SBATCH --job-name=dqn_pixel
#SBATCH --partition=1080ti-short
#SBATCH --time=00-01:00:00
#SBATCH --mem=4000
#SBATCH --account=rkozma
#SBATCH --output=res_%j.txt
#SBATCH -e res_%j.err
#SBATCH --gres=gpu:1

python3 dqn_ann_0.05.py
exit