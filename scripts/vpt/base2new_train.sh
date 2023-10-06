#!/bin/bash

cd ../..

# custom config
DATA=~/projects/prob-cocoop/data
TRAINER=VPT

DATASET=$1
SEED=$2
GPUIDS=$3
L=$4
EPOCHS=$5

CFG=vit_b16_c4_ep10_batch1_ctxv1
SHOTS=16



DIR=output/base2new/train_base/${DATASET}/mcmc_${L}_epochs_${EPOCHS}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
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
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base \
    OPTIM.MAX_EPOCH ${EPOCHS} \
    TRAINER.VPT.L ${L}
fi
