#!/bin/bash

model_path=meta-llama/Llama-3.2-3B-Instruct

python -u generate_freeze.py --model_path $model_path --freeze_type attn_vals --dataset jailbreakbench
python -u generate_freeze.py --model_path $model_path --freeze_type attn_vals --dataset alpaca
python -u generate_freeze.py --model_path $model_path --freeze_type mlps --dataset jailbreakbench
python -u generate_freeze.py --model_path $model_path --freeze_type mlps --dataset alpaca
# python -u generate_freeze.py --model_path $model_path --freeze_type attn_weights --dataset jailbreakbench
# python -u generate_freeze.py --model_path $model_path --freeze_type attn_weights --dataset alpaca
# python -u generate_freeze.py --model_path $model_path --freeze_type iic --dataset jailbreakbench
# python -u generate_freeze.py --model_path $model_path --freeze_type iic --dataset alpaca
# python -u generate_freeze.py --model_path $model_path --freeze_type mlp_direct --dataset jailbreakbench
# python -u generate_freeze.py --model_path $model_path --freeze_type mlp_direct --dataset alpaca