#!/bin/sh
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --account=kunwarsingh
#SBATCH --job-name=horovod_test
#SBATCH --nodelist=gpu009
#SBATCH --error=job.%A_%a.err
#SBATCH --output=job.%A_%a.out

# Initializing Shell
conda init bash
source ~/.bashrc

# Loading Modules
module load openmpi/openmpi_4.1.2 cuda/10.2 cmake/3.26.4 DL/DL-CondaPy/3.7 compiler/gcc/8.3.0
module unload openmpi/openmpi_4.1.2
module load openmpi/openmpi_4.1.2

# Activating Conda
conda activate horovod-editable

# Running Horovod
mpirun -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca btl ^openib -np 1 -host gpu009:1 python train.py --configs configs/cifar/resnet110.py configs/dgc/wm5.py configs/dgc/fp16.py configs/dgc/int32.py --configs.train.optimize_bn_separately 1 > results/cifar-n1-9.txt
mpirun -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca btl ^openib -np 1 -host gpu009:1 python train.py --configs configs/cifar/resnet110.py --configs.train.optimize_bn_separately 1 --configs.train.dgc False > results/cifar-n1-9-nodgc.txt