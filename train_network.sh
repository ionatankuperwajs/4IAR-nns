#!/bin/bash
#SBATCH --job-name=4IAR_nn
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ik1125@nyu.edu
#SBATCH --output=4IAR_nn.out
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=8GB

# For linear networks
# python load_train.py -m 'linear' -v 1 -hl 1 -u 200 -b 12 -e 10 -lr .001 -d #overlay_ext3

# For convolutional networks
# python load_train.py -m 'conv' -v 1 -hl 1 -f 4 -fs 3 -s 1 -p 1 -b 12 -e 10 -lr .001 -d $overlay_ext3

overlay_ext3=/nn_peakdata/train

singularity \
    exec --nv $(for sqf in /scratch/ik1125/nn_peakdata/*.sqf; do echo --overlay $sqf:ro; done) \
    --overlay /scratch/ik1125/overlay-50G-10M.ext3 \
    /scratch/work/public/singularity/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif \
    /bin/bash -c "source /home/ik1125/.bashrc;
		 conda activate /ext3/4IAR-conda; \
                  python load_train.py -m 'linear' -v 1 -hl 1 -u 200 -b 12 -e 10 -lr .001 -d $overlay_ext3"
