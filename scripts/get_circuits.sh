#!/bin/bash

model=${model:-"Qwen/Qwen3-8B"}
layer=${layer:-20}
learn_type=${learn:-"dim"}
learn_path=${lp:-'none'}
method=${method:-'ig2'}

echo "experiments with $model at layer $layer"

echo "act patching logit with base harmful"
python act_patching.py --model_path $model \
    --exp_name "${method}_${learn_type}_logit_base_harmful" \
    --method $method \
    --metric logit --layer $layer --learn_type $learn_type --learn_path $learn_path  \
    --harm_flag

echo "act patching logit with base harmless"
python act_patching.py --model_path $model \
    --exp_name "${method}_${learn_type}_logit_base_harmless" \
    --method $method \
    --metric logit --layer $layer --learn_type $learn_type --learn_path $learn_path 

echo "act patching logit with steer harmless"
python act_patching.py --model_path $model \
    --exp_name "${method}_${learn_type}_logit_steer_harmless" \
    --method $method \
    --metric logit --layer $layer --learn_type $learn_type --learn_path $learn_path  \
    --steer_flag

echo "act patching logit with steer harmful"
python act_patching.py --model_path $model \
    --exp_name "${method}_${learn_type}_logit_steer_harmful" \
    --method $method \
    --metric logit --layer $layer --learn_type $learn_type --learn_path $learn_path  \
    --harm_flag --steer_flag

echo "aggregating patching results"
python patching/aggregate_patching.py \
    ${method}_${learn_type}_logit_steer_harmless ${method}_${learn_type}_logit_steer_harmful ${method}_${learn_type}_logit_base_harmless ${method}_${learn_type}_logit_base_harmful \
    --model_path $model \
    --save_name ${method}_${learn_type}_logit

