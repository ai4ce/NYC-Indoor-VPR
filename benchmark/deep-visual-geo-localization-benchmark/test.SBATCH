#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=64GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu
#SBATCH --job-name=res

module purge

singularity exec --nv \
	    --overlay /scratch/ds5725/environments/mixvpr.ext3:rw \
	    /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif \
	    /bin/bash -c "source /ext3/env.sh; python train_1.py --dataset_name=indoor --datasets_folder=/mnt/data/nyc_indoor --backb"
