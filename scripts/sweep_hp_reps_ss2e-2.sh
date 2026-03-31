#!/bin/bash

model_path=${1:-google/gemma-2-2b-it}
layer=${2:-15}
seed=${3:-42}

python -u learn_sv.py --model_path $model_path --exp_name "reps_bsz6_lr1e-2_ss2e-2_seed$seed" --intervention_layer $layer \
    --loss_type reps \
    --batch_size 6 \
    --mini_batch_size 1 \
    --lr 0.01 --seed $seed --pbar
python -u learn_sv.py --model_path $model_path --exp_name "reps_bsz6_lr4e-2_ss2e-2_seed$seed" --intervention_layer $layer \
    --loss_type reps \
    --batch_size 6 \
    --mini_batch_size 1 \
    --lr 0.04 --seed $seed
python -u learn_sv.py --model_path $model_path --exp_name "reps_bsz12_lr1e-2_ss2e-2_seed$seed" --intervention_layer $layer \
    --loss_type reps \
    --batch_size 12 \
    --mini_batch_size 1 \
    --lr 0.01 --seed $seed
python -u learn_sv.py --model_path $model_path --exp_name "reps_bsz12_lr4e-2_ss2e-2_seed$seed" --intervention_layer $layer \
    --loss_type reps \
    --batch_size 12 \
    --mini_batch_size 1 \
    --lr 0.04 --seed $seed