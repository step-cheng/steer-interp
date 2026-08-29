#!/bin/bash

model_path=${model:-"Qwen/Qwen3-8B"}
layer=${layer:-20}
seed=${seed:-42}

python -u learn_sv.py --model_path $model_path --exp_name "ntp_bsz6_lr1e-2_seed$seed" --intervention_layer $layer \
    --loss_type ntp \
    --batch_size 6 \
    --lr 0.01 --seed $seed
python -u learn_sv.py --model_path $model_path --exp_name "ntp_bsz6_lr4e-2_seed$seed" --intervention_layer $layer \
    --loss_type ntp \
    --batch_size 6 \
    --lr 0.04 --seed $seed
python -u learn_sv.py --model_path $model_path --exp_name "ntp_bsz12_lr1e-2_seed$seed" --intervention_layer $layer \
    --loss_type ntp \
    --batch_size 12 \
    --lr 0.01 --seed $seed
python -u learn_sv.py --model_path $model_path --exp_name "ntp_bsz12_lr4e-2_seed$seed" --intervention_layer $layer \
    --loss_type ntp \
    --batch_size 12 \
    --lr 0.04 --seed $seed
