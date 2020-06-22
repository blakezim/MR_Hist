#!/bin/bash

#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --job-name="Hist"
#SBATCH --mem=128G
#SBATCH -t 2-00:00 # time (D-HH:MM)


source /home/sci/blakez/software/anaconda3/bin/activate MR_Hist
type python

echo "Running with GPU Index $CUDA_VISIBLE_DEVICES"

python -u ../histology_seg.py  --threads 4 -r 400 --trainBatchSize 100 --inferBatchSize 64 --nEpochs 1000;
