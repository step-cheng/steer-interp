#!/bin/bash

model_path=google/gemma-2-2b-it
python -u learn_sv.py --model_path $model_path --exp_name ntp_bsz6_lr1e-2 \
    --load_pretrained --val_batch_size 2
python -u learn_sv.py --model_path $model_path --exp_name ntp_bsz6_lr4e-2 \
    --load_pretrained --val_batch_size 2
python -u learn_sv.py --model_path $model_path --exp_name ntp_bsz12_lr1e-2 \
    --load_pretrained --val_batch_size 2
python -u learn_sv.py --model_path $model_path --exp_name ntp_bsz12_lr4e-2 \
    --load_pretrained --val_batch_size 2

python -u learn_sv.py --model_path $model_path --exp_name reps_bsz6_lr1e-2_ss2e-2 \
    --load_pretrained --val_batch_size 2
python -u learn_sv.py --model_path $model_path --exp_name reps_bsz6_lr4e-2_ss2e-2 \
    --load_pretrained --val_batch_size 2
python -u learn_sv.py --model_path $model_path --exp_name reps_bsz12_lr1e-2_ss2e-2 \
    --load_pretrained --val_batch_size 2
python -u learn_sv.py --model_path $model_path --exp_name reps_bsz12_lr4e-2_ss2e-2 \
    --load_pretrained --val_batch_size 2

python -u learn_sv.py --model_path $model_path --exp_name reps_bsz6_lr1e-2_ss1e-5 \
    --load_pretrained --val_batch_size 2
python -u learn_sv.py --model_path $model_path --exp_name reps_bsz6_lr4e-2_ss1e-5 \
    --load_pretrained --val_batch_size 2
python -u learn_sv.py --model_path $model_path --exp_name reps_bsz12_lr1e-2_ss1e-5 \
    --load_pretrained --val_batch_size 2
python -u learn_sv.py --model_path $model_path --exp_name reps_bsz12_lr4e-2_ss1e-5 \
    --load_pretrained --val_batch_size 2