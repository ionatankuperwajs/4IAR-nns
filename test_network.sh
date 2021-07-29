#!/bin/bash
#SBATCH --job-name=4IAR_nn
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ik1125@nyu.edu
#SBATCH --output=slurm_output/4IAR_test_1.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --time=24:00:00
#SBATCH --mem=16GB

singularity \
    exec $(for sqf in /scratch/ik1125/nn_peakdata/*.sqf; do echo --overlay $sqf:ro; done) \
    --overlay /scratch/ik1125/overlay-50G-10M.ext3:ro \
    /scratch/work/public/singularity/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif \
    /bin/bash -c "source /home/ik1125/.bashrc;
		 conda activate /ext3/4IAR-conda; \
                  python load_test.py -m 'linearskip' -v 1 -b 128 "
