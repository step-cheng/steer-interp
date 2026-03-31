#!/bin/bash

model_path=${1:-"google/gemma-2-2b-it"}
layer=${2:-15}
seed=${3:-5}
dim_ref_vector_path="pipeline/runs/gemma-2-2b-it/direction_layer15_pos-1.pt"
ntp_ref_vector_path="trained_vectors/gemma-2-2b-it/ntp_bsz6_lr1e-2_seed42/direction_8.pt"
reps_ref_vector_path="trained_vectors/gemma-2-2b-it/reps_bsz6_lr1e-2_ss2e-2_seed5/direction_1.pt"

python -u learn_sv.py --model_path $model_path --exp_name "ortho_bsz6_lr1e-2_seed$seed" --intervention_layer $layer \
    --loss_type ortho \
    --batch_size 6 \
    --ref_vector_paths $dim_ref_vector_path $ntp_ref_vector_path $reps_ref_vector_path \
    --lr 0.01 --seed $seed
python -u learn_sv.py --model_path $model_path --exp_name "ortho_bsz6_lr4e-2_seed$seed" --intervention_layer $layer \
    --loss_type ortho \
    --batch_size 6 \
    --ref_vector_paths $dim_ref_vector_path $ntp_ref_vector_path $reps_ref_vector_path \
    --lr 0.04 --seed $seed
python -u learn_sv.py --model_path $model_path --exp_name "ortho_bsz12_lr1e-2_seed$seed" --intervention_layer $layer \
    --loss_type ortho \
    --batch_size 12 \
    --ref_vector_paths $dim_ref_vector_path $ntp_ref_vector_path $reps_ref_vector_path  \
    --lr 0.01 --seed $seed
python -u learn_sv.py --model_path $model_path --exp_name "ortho_bsz12_lr4e-2_seed$seed" --intervention_layer $layer \
    --loss_type ortho \
    --batch_size 12 \
    --ref_vector_paths $dim_ref_vector_path $ntp_ref_vector_path $reps_ref_vector_path  \
    --lr 0.04 --seed $seed
