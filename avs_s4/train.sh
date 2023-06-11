#!/bin/bash

#SBATCH --job-name test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=3G
#SBATCH --time 1-0
#SBATCH --partition batch_ce_ugrad
#SBATCH -x sw7
#SBATCH --array 2-7%1
#SBATCH -o slurm/logs/slurm-%A-%x.out

setting='S4'
visual_backbone="resnet" # "resnet" or "pvt"

#spring.submit arun --gpu -n1 --gres=gpu:1 --quotatype=auto -p MMG --job-name="train_${setting}_${visual_backbone}" \

python train.py \
        --session_name ${setting}_${visual_backbone} \
        --visual_backbone ${visual_backbone} \
        --train_batch_size 4 \
        --lr 0.0001 \
        --tpavi_stages 0 1 2 3 \
        --tpavi_va_flag \
        --max_epoches 1 \
