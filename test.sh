#!/bin/bash
#SBATCH --job-name=vlm-test
#SBATCH --gres=gpu:t4:4
#SBATCH --qos=m3
#SBATCH --time=3:00:00
#SBATCH -c 30
#SBATCH --mem=60G
#SBATCH --output=/scratch/ssd004/scratch/murdock/physics-vlm/outs/output/slurm-%j.out
#SBATCH --error=/scratch/ssd004/scratch/murdock/physics-vlm/outs/error/slurm-%j.err

cd /scratch/ssd004/scratch/murdock/physics-vlm

srun python data.py