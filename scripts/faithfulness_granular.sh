#!/bin/bash

model=${1:-"google/gemma-2-2b-it"}
IFS1=' ' read -r -a n_list <<< "${2:-"100 200 500 1000 1500"}"
IFS2=' ' read -r -a exp_names <<< "${3:-"dim_logit dim_dkl0"}"
learn_type=${4-"dim"}
method=${5:-"simple"}
learn_path=${6:-None}
prefix_dir=${7:-None}
invert=${8}

invert_flag=""
if [ "$invert" = "invert" ]; then
    invert_flag="--invert"
fi

for exp_name in "${exp_names[@]}"; do
    for n in "${n_list[@]}"; do
        python evaluate_circuit.py --exp_name $exp_name --n $n --model_path $model --method $method --granular
        python faithfulness_granular.py --exp_name $exp_name --model_path $model --n $n \
            --learn_type $learn_type --learn_path $learn_path --prefix_dir $prefix_dir --method $method $invert_flag
        python faithfulness_granular.py --exp_name $exp_name --model_path $model --n $n --harm_flag \
            --learn_type $learn_type --learn_path $learn_path --prefix_dir $prefix_dir --method $method $invert_flag
    done
done