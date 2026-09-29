# What Drives Representation Steering? A Mechanistic Case Study on Steering Refusal

This repository accompanies the paper "What Drives Representations Steering? A Mechanistic Case Study on Steering Refusal". For reproducibility, we provide code to reproduce the main results, as well as data from the original results.

Paper: https://arxiv.org/abs/2604.08524

## Setup

To download requirements:
We use Python 3.11. To install dependencies, create a conda environment with
```
conda create --name <env> --file requirements.txt
```

## Reproducing Main Results

We adapt code from https://github.com/andyrdt/refusal_direction to generate steered and unsteered rollouts on harmful and harmless questions. We adapt code from https://github.com/saprmarks/feature-circuits to perform attribution patching.
Below are instructions to reproduce results for a model, such as meta-llama/Llama-3.2-3B-Instruct.

### Difference-in-Means Vector Pipeline
To generate steered and unsteered rollouts on harmless and harmful questions for a model and obtain the best Difference-in-Means vector:
```
python -m refusal_direction.pipeline.run_pipeline --model_path meta-llama/Llama-3.2-3B-Instruct
```

From our paper, the best layer for meta-llama/Llama-3.2-3B-Instruct is layer 12. Prepare the attribution patching data for the Difference in Means vector:
```
model=meta-llama/Llama-3.2-3B-Instruct layer=12 learn_type=dim bash scripts/data_preprocess.sh
```

Perform patching on a model with the generated rollouts, run:
```
model=meta-llama/Llama-3.2-3B-Instruct layer=12 learn_type=dim bash scripts/get_circuits.sh
```

To generate circuits at incremental sizes and test their faithfulness:
```
model=meta-llama/Llama-3.2-3B-Instruct bash scripts/faithfulness.sh
```

### Other Vectors
The paper also includes results for steering vectors learned through next token prediction and peference optimization, as well as sparse steering vectors. Apart from the method in which these vectors are obtained, the steps for attribution patching and evaluating circuit faithfulness the same. 
To obtain sparse steering vectors.
```
python obtain_sparse_steer_vec.py --model_path meta-llama/Llama-3.2-3B-Instruct --save --exp_name <exp_name>
```

Learn steering vectors via next token prediction (ntp) and preference optimization (reps):
```
python generate_train_data.py --model meta-llama/Llama-3.2-3B-Instruct
model=meta-llama/Llama-3.2-3B-Instruct layer=12 bash scripts/sweep_hp_ntp.sh
model=meta-llama/Llama-3.2-3B-Instruct layer=12 bash scripts/sweep_hp_reps.sh
```

### Steering Value Vector Inspection
The steering value vector is the input-independent, direct effect that the steering vector has on the attention mechanism's value projection. These value vectors decode to semantically interpretable concepts with vocabulary projection. Code for the svv, and an example is in svv_node.py:
```
python svv_node.py --model_path meta-llama/Llama-3.2-3B-Instruct --learn_type dim
```


### Citing this work
If you find this work useful in your research, consider citing our paper:

```
@misc{cheng2026drivesrepresentationsteeringmechanistic,
      title={What Drives Representation Steering? A Mechanistic Case Study on Steering Refusal}, 
      author={Stephen Cheng and Sarah Wiegreffe and Dinesh Manocha},
      year={2026},
      eprint={2604.08524},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2604.08524}, 
}
```
