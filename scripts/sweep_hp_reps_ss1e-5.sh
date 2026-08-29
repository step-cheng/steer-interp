#!/bin/bash

model_path=${model:-google/gemma-2-2b-it}
layer=${layer:-20}
seed=${seed:-42}

python -u learn_sv.py --model_path $model_path --exp_name "reps_bsz6_lr1e-2_ss1e-5_seed$seed" --intervention_layer $layer \
    --loss_type reps \
    --batch_size 6 \
    --lr 0.01 \
    --simpo_scaler 1e-5 --seed $seed
python -u learn_sv.py --model_path $model_path --exp_name "reps_bsz6_lr4e-2_ss1e-5_seed$seed" --intervention_layer $layer \
    --loss_type reps \
    --batch_size 6 \
    --lr 0.04 \
    --simpo_scaler 1e-5 --seed $seed
python -u learn_sv.py --model_path $model_path --exp_name "reps_bsz12_lr1e-2_ss1e-5_seed$seed" --intervention_layer $layer \
    --loss_type reps \
    --batch_size 12 \
    --lr 0.01 \
    --simpo_scaler 1e-5 --seed $seed
python -u learn_sv.py --model_path $model_path --exp_name "reps_bsz12_lr4e-2_ss1e-5_seed$seed" --intervention_layer $layer \
    --loss_type reps \
    --batch_size 12 \
    --lr 0.04 \
    --simpo_scaler 1e-5 --seed $seed
