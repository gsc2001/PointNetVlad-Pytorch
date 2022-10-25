#!/bin/bash
#SBATCH -A rrc
#SBATCH --reservation rrc
#SBATCH -c 10
#SBATCH -w gnode043
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --mail-user=gurkirat.singh@students.iiit.ac.in
#SBATCH --mail-type=ALL
module add cuda/9.0
module add cudnn/7-cuda-9.0

nvidia-smi

echo "Activating conda"

source ~/miniconda3/bin/activate convex-net
python --version

echo "Moving code"

cp ~/gsc2001/ConvexNet /scratch -rf
echo "Running script"

python /scratch/ConvexNet/src/train.py

echo "Done"


