#!/usr/bin/env bash
#
#SBATCH --job-name=benchmarks
#SBATCH --partition=1080ti-short
#SBATCH --time=00-4:00:00
#SBATCH --mem=8000
#SBATCH --account=rkozma
#SBATCH --output=res_%j.txt
#SBATCH -e res_%j.err
#SBATCH --gres=gpu:1


seed=${1:-0}
num_episodes=${2:-1}

echo $seed $num_episodes
python3 dqn_snn_0.05.py @@seed $seed @@num_episodes $num_episodes
exit