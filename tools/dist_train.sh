#!/usr/bin/env bash
#./tools/dist_train.sh ./projects/configs/surroundocc/surroundocc_4d.py 8  ./work_dirs/surroundocc
CONFIG=$1
GPUS=1
SAVE_PATH=$3
PORT=${PORT:-28108}
NCCL_DEBUG=INFO

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --work-dir ${SAVE_PATH} --resume-from ./ckpts/surroundocc_semantic.pth  --launcher pytorch ${@:4} --deterministic
