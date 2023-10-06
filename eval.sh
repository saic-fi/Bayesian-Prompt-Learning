dataset=$1
mcmc=$2
epochs=$3

python parse_test_res.py ./ls/base2new/train_base/${dataset}/mcmc_${mcmc}_epochs_${epochs}/shots_16/CoCoOp/vit_b16_c4_ep10_batch1_ctxv1/
python parse_test_res.py ./ls/base2new/test_new/${dataset}/mcmc_${mcmc}_epochs_${epochs}/shots_16/CoCoOp/vit_b16_c4_ep10_batch1_ctxv1/ --test-log
