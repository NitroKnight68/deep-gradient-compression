#!/bin/bash
export LD_LIBRARY_PATH=/home/rithvik/8th_sem/nccl_2.15.1-1+cuda10.2_x86_64/lib
export HOROVOD_NCCL_HOME=/home/rithvik/8th_sem/nccl_2.15.1-1+cuda10.2_x86_64
export HOROVOD_GPU_OPERATIONS=NCCL
export HOROVOD_WITH_PYTORCH=1
export HOROVOD_WITHOUT_TENSORFLOW=1
export HOROVOD_WITHOUT_MXNET=1
export HOROVOD_WITH_GLOO=1
export HOROVOD_WITH_MPI=1