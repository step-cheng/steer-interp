#!/bin/bash

model=${model:-"Qwen/Qwen3-8B"}
layer=${layer:-20}
learn_type=${learn:-"dim"}
learn_path=${lp:-'none'}
method=${method:-'ig2'}

echo "experiments with $model at layer $layer"

# echo "act patching logit with base harmful"
# python act_patching.py --model_path $model \
#     --exp_name "${method}_${learn_type}_logit_base_harmful" \
#     --method ig2 \
#     --metric logit --layer $layer --learn_type $learn_type --learn_path $learn_path  \
#     --harm_flag

# echo "act patching logit with base harmless"
# python act_patching.py --model_path $model \
#     --exp_name "${method}_${learn_type}_logit_base_harmless" \
#     --method ig2 \
#     --metric logit --layer $layer --learn_type $learn_type --learn_path $learn_path 

# echo "act patching logit with steer harmless"
# python act_patching.py --model_path $model \
#     --exp_name "${method}_${learn_type}_logit_steer_harmless" \
#     --method ig2 \
#     --metric logit --layer $layer --learn_type $learn_type --learn_path $learn_path  \
#     --steer_flag

echo "act patching logit with steer harmful"
python act_patching.py --model_path $model \
    --exp_name "${method}_${learn_type}_logit_steer_harmful" \
    --method ig2 \
    --metric logit --layer $layer --learn_type $learn_type --learn_path $learn_path  \
    --harm_flag --steer_flag


python patching/aggregate_patching.py \
    ig2_dim_logit_steer_harmless ig2_dim_logit_steer_harmful ig2_dim_logit_base_harmless ig2_dim_logit_base_harmful \
    --model_path Qwen/Qwen3-8B \
    --save_nam ig2_dim_logit


# echo "act patching dkl with base harmless"
# python act_patching.py --model_path $model \
#     --exp_name "${learn_type}_dkl0_base_harmless" \
#     --metric dirKL --layer $layer --learn_type $learn_type --learn_path $learn_path  

# echo "act patching dkl with base harmful"
# python act_patching.py --model_path $model \
#     --exp_name "${learn_type}_dkl0_base_harmful" \
#     --metric dirKL --layer $layer --learn_type $learn_type --learn_path $learn_path  \
#     --harm_flag

# echo "act patching dkl with steer harmless"
# python act_patching.py --model_path $model \
#     --exp_name "${learn_type}_dkl0_steer_harmless" \
#     --metric dirKL --layer $layer --learn_type $learn_type --learn_path $learn_path  \
#     --steer_flag

# echo "act patching dkl with steer harmful"
# python act_patching.py --model_path $model \
#     --exp_name "${learn_type}_dkl0_steer_harmful" \
#     --metric dirKL --layer $layer --learn_type $learn_type --learn_path $learn_path  \
#     --harm_flag --steer_flag

# python evaluate_circuit.py --exp_name "${ig2}_${learn_type}_logit_base_harmless" --model_path $model \
#     --n 50 --plot --label_vals 
# python evaluate_circuit.py --exp_name "${ig2}_${learn_type}_logit_base_harmful" --model_path $model \
#     --n 50 --plot --label_vals
# python evaluate_circuit.py --exp_name "${ig2}_${learn_type}_logit_steer_harmful" --model_path $model \
#     --n 50 --plot --label_vals
# python evaluate_circuit.py --exp_name "${ig2}_${learn_type}_logit_steer_harmless" --model_path $model \
#     --n 50 --plot --label_vals
# python evaluate_circuit.py --exp_name "${learn_type}_dkl0_base_harmless" --model_path $model \
#     --n 50 --plot --label_vals
# python evaluate_circuit.py --exp_name "${learn_type}_dkl0_base_harmful" --model_path $model \
#     --n 50 --plot --label_vals
# python evaluate_circuit.py --exp_name "${learn_type}_dkl0_steer_harmful" --model_path $model \
#     --n 50 --plot --label_vals
# python evaluate_circuit.py --exp_name "${learn_type}_dkl0_steer_harmless" --model_path $model \
#     --n 50 --plot --label_vals
