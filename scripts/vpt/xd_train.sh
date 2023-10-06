#!/bin/bash

cd ../..

# custom config
DATA=~/projects/prob-cocoop/data
TRAINER=VPT

DATASET=imagenet
SEED=$1
L=$2
EPOCHS=$3

CFG=vit_b16_c4_ep10_batch1_ctxv1
SHOTS=16


DIR=output/${DATASET}/mcmc_${L}_epochs_${EPOCHS}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
else
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    OPTIM.MAX_EPOCH ${EPOCHS} \
    TRAINER.VPT.L ${L}
fi
