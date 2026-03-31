#!/bin/bash

model=${1:-"meta-llama/Llama-3.2-3B-Instruct"}
layer=${2:-12}
learn_type=${3:-"dim"}
learn_path=${4}

echo "experiments with $model at layer $layer"

echo "preparing data for circuit discovery"
# python data_preprocess.py --model_path $model --layer $layer --harm_flag
# python data_preprocess.py --model_path $model --layer $layer --steer_flag
# python data_preprocess.py --model_path $model --layer $layer --harm_flag --steer_flag
# python data_preprocess.py --model_path $model --layer $layer

echo "act patching logit with base harmful"
python act_patching.py --model_path $model \
    --exp_name "${learn_type}_logit_base_harmful" \
    --metric logit --layer $layer --learn_type $learn_type --learn_path $learn_path  \
    --harm_flag

echo "act patching logit with base harmless"
python act_patching.py --model_path $model \
    --exp_name "${learn_type}_logit_base_harmless" \
    --metric logit --layer $layer --learn_type $learn_type --learn_path $learn_path 

echo "act patching logit with steer harmless"
python act_patching.py --model_path $model \
    --exp_name "${learn_type}_logit_steer_harmless" \
    --metric logit --layer $layer --learn_type $learn_type --learn_path $learn_path  \
    --steer_flag

echo "act patching logit with steer harmful"
python act_patching.py --model_path $model \
    --exp_name "${learn_type}_logit_steer_harmful" \
    --metric logit --layer $layer --learn_type $learn_type --learn_path $learn_path  \
    --harm_flag --steer_flag

echo "act patching dkl with base harmless"
python act_patching.py --model_path $model \
    --exp_name "${learn_type}_dkl0_base_harmless" \
    --metric dirKL --layer $layer --learn_type $learn_type --learn_path $learn_path  

echo "act patching dkl with base harmful"
python act_patching.py --model_path $model \
    --exp_name "${learn_type}_dkl0_base_harmful" \
    --metric dirKL --layer $layer --learn_type $learn_type --learn_path $learn_path  \
    --harm_flag

echo "act patching dkl with steer harmless"
python act_patching.py --model_path $model \
    --exp_name "${learn_type}_dkl0_steer_harmless" \
    --metric dirKL --layer $layer --learn_type $learn_type --learn_path $learn_path  \
    --steer_flag

echo "act patching dkl with steer harmful"
python act_patching.py --model_path $model \
    --exp_name "${learn_type}_dkl0_steer_harmful" \
    --metric dirKL --layer $layer --learn_type $learn_type --learn_path $learn_path  \
    --harm_flag --steer_flag

python evaluate_circuit.py --exp_name "${learn_type}_dkl0_base_harmless" --model_path $model \
    --n 50 --plot --label_vals
python evaluate_circuit.py --exp_name "${learn_type}_dkl0_base_harmful" --model_path $model \
    --n 50 --plot --label_vals
python evaluate_circuit.py --exp_name "${learn_type}_dkl0_steer_harmful" --model_path $model \
    --n 50 --plot --label_vals
python evaluate_circuit.py --exp_name "${learn_type}_dkl0_steer_harmless" --model_path $model \
    --n 50 --plot --label_vals
python evaluate_circuit.py --exp_name "${learn_type}_logit_base_harmless" --model_path $model \
    --n 50 --plot --label_vals 
python evaluate_circuit.py --exp_name "${learn_type}_logit_base_harmful" --model_path $model \
    --n 50 --plot --label_vals
python evaluate_circuit.py --exp_name "${learn_type}_logit_steer_harmful" --model_path $model \
    --n 50 --plot --label_vals
python evaluate_circuit.py --exp_name "${learn_type}_logit_steer_harmless" --model_path $model \
    --n 50 --plot --label_vals
