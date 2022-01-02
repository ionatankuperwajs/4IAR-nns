#!/bin/bash
#SBATCH --job-name=4IAR_nn
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hhs4@nyu.edu
#SBATCH --output=slurm_output/4IAR_nn_%a.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --time=168:00:00
#SBATCH --mem=40GB

# For linear networks
# python load_train.py -m 'linear' -v 1 -hl 1 -u 200 -b 12 -e 10 -lr .001 -d #overlay_ext3

# For convolutional networks
# python load_train.py -m 'conv' -v 1 -hl 1 -f 4 -fs 3 -s 1 -p 1 -b 12 -e 10 -lr .001 -d $overlay_ext3

# overlay_ext3=/nn_peakdata

a=$SLURM_ARRAY_TASK_ID
array_u0=(500 1000 2000 4000 8000)
array_hl0=(5 10 20 40 80)
array_u=()
array_hl=()
for u in "${array_u0[@]}"
do
  for hl in "${array_hl0[@]}"
  do
    array_u+=($u)
    array_hl+=($hl)
  done
done
echo $a
echo ${array_u[$a]}
echo ${array_hl[$a]}

singularity \
   exec $(for sqf in /scratch/ik1125/nn_peakdata/*; do echo --overlay $sqf:ro; done) \
   --overlay /scratch/ik1125/overlay-50G-10M.ext3:ro \
   /scratch/work/public/singularity/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif \
   /bin/bash -c "source /home/hhs4/.bashrc;
	         conda activate /ext3/4IAR-conda; \
                 python load_train.py -m 'linearskip' -v $a -hl ${array_hl[$a]} -u ${array_u[$a]} -bn 50 -b 128 -e 10 -lr .001 -c"
