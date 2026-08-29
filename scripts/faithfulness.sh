#!/bin/bash

model=${model:-"google/gemma-2-2b-it"}
IFS1=' ' read -r -a n_list <<< "${ns:-"100 200 500 1000 1500"}"
IFS2=' ' read -r -a exp_names <<< "${exp_names:-"dim_logit dim_dkl0"}"
learn_type=${learn}
method=${method:-simple}
learn_path=${lp:-None}
prefix_dir=${prefix_dir:-None}

for exp_name in "${exp_names[@]}"; do
    for n in "${n_list[@]}"; do
        python evaluate_circuit.py --exp_name $exp_name --n $n --model_path $model --method $method
        python faithfulness.py --exp_name $exp_name --model_path $model --n $n \
            --learn_type $learn_type --learn_path $learn_path --prefix_dir $prefix_dir --method $method --invert
        python faithfulness.py --exp_name $exp_name --model_path $model --n $n --harm_flag \
            --learn_type $learn_type --learn_path $learn_path --prefix_dir $prefix_dir --method $method --invert
    done
done