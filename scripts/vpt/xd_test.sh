#!/bin/bash

cd ../..

# custom config
DATA=~/projects/prob-cocoop/data
TRAINER=VPT

DATASET=$1
SEED=$2
L=$3
EPOCHS=$4
LOADEP=$4
GPUIDS=$5

CFG=vit_b16_c4_ep10_batch1_ctxv1
SHOTS=16


DIR=output/evaluation/${TRAINER}/${CFG}_${SHOTS}shots/${DATASET}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
else
    CUDA_VISIBLE_DEVICES=${GPUIDS} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir output/imagenet/mcmc_${L}_epochs_${EPOCHS}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED} \
    --load-epoch ${LOADEP} \
    --eval-only \
    OPTIM.MAX_EPOCH ${EPOCHS} \
    TRAINER.VPT.L ${L}
fi
