#!/bin/bash

model=${1:-"meta-llama/Llama-3.2-3B-Instruct"}
layer=${2:-12}
learn_type=${3:-"dim"}
learn_path=${4:-None}

echo "preparing data for circuit discovery"
python data_preprocess.py --model_path $model --layer $layer \
    --learn_type $learn_type --learn_path $learn_path 

python data_preprocess.py --model_path $model --layer $layer \
    --learn_type $learn_type --learn_path $learn_path \
    --harm_flag

python data_preprocess.py --model_path $model --layer $layer \
    --learn_type $learn_type --learn_path $learn_path \
    --steer_flag

python data_preprocess.py --model_path $model --layer $layer \
    --learn_type $learn_type --learn_path $learn_path \
    --harm_flag --steer_flag