#!/bin/bash
#SBATCH --partition=rtx6000
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=960
#SBATCH --job-name=vlm-script
#SBATCH --output=/scratch/ssd004/scratch/murdock/physics-vlm/outs/output/slurm-%j.out
#SBATCH --error=/scratch/ssd004/scratch/murdock/physics-vlm/outs/error/slurm-%j.err

cd /scratch/ssd004/scratch/murdock/physics-vlm

srun python finetune_lora.py