#!/bin/bash
#SBATCH -A rrc
#SBATCH --reservation rrc
#SBATCH -c 10
#SBATCH -w gnode043
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=gurkirat.singh@students.iiit.ac.in
#SBATCH --mail-type=ALL
nvidia-smi

echo "Activating conda"

source ~/miniconda3/bin/activate torch_pointnetvlad

python --version
python ~/RRC/collabslam/repos/PointNetVlad-Pytorch --batch_num_queries 2 --log_dir /scratch/gurkirat.singh/training_logs/pointnetvlad/logs/

echo "Done"


