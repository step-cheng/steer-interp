import os
import pandas as pd
import json
import torch as t

from transformers import AutoModelForCausalLM, AutoTokenizer

from .gen_utils import PROJECT_DIR

def load_refusal_completions(model_path, steer_flag, concept_params: dict, learned: bool, pos: int, layer: int, vector_base_path: str = None, num=100):
    harm_flag = concept_params['harm_flag']

    model_name = model_path.split('/')[-1]
    if learned:
        suffix = f"_{vector_base_path}"
    else:
        suffix = f"_layer{layer}_pos{pos}"
    dir_path = f'refusal_direction/pipeline/runs/{model_name}/completions{suffix}'
    print(f'loading data from {dir_path}')
    
    dataset_name = "jailbreakbench" if harm_flag else "harmless"
    f_name = "{dataset_name}_{steer_name}_evaluations.json"

    with open(os.path.join(dir_path, f_name.format(dataset_name=dataset_name, steer_name="actadd"))) as f:
        evals_steer = json.load(f)

    with open(os.path.join(dir_path, f_name.format(dataset_name=dataset_name, steer_name="baseline"))) as f:
        evals_base = json.load(f)
    
    completions_steer = evals_steer['completions']
    completions_base = evals_base['completions']
    if not harm_flag:
        # testing alpaca, hacky way to keep only 100 for patching exps
        completions_steer = completions_steer[:num]
        completions_base = completions_base[:num]
    contents = []

    if harm_flag:
        for comp_steer, comp_base in zip(completions_steer, completions_base):
            if comp_steer['is_jailbreak_llamaguard2'] == 1 and comp_base['is_jailbreak_llamaguard2'] == 0:
                comp_to_add = comp_steer if steer_flag else comp_base
                contents.append([
                    {"role": "user", "content": comp_to_add['prompt']},
                    {"role": "assistant", "content": comp_to_add['response']}
                ])
    else:
        for comp_steer, comp_base in zip(completions_steer, completions_base):
            if comp_steer['is_jailbreak_substring_matching'] == 0 and comp_base['is_jailbreak_substring_matching'] == 1:
                comp_to_add = comp_steer if steer_flag else comp_base
                contents.append([
                    {"role": "user", "content": comp_to_add['prompt']},
                    {"role": "assistant", "content": comp_to_add['response']}
                ])
    print(f'len contents: {len(contents)}')

    return contents

def load_test_prompt():
    content = [
        {"role": "user", "content": "How do I take care of the sick and elderly?"},
        {"role": "assistant", "content": "I cannot provide information on how to take care of the sick or elderly. Can I help you with something else?"}
    ]
    return [content]

def _load_refusal_dim_vector(concept: str, model_path: str, concept_params: dict, pos: int, layer_idx: int, check_best=False):
    harm_flag = concept_params['harm_flag']

    suffix = f"_layer{layer_idx}_pos{pos}"
    # if check_best:
    #     base_path = f"pipeline/runs/{model_path.split('/')[-1]}"
    #     f_name_steer = f"direction{suffix}.pt"
    #     f_name_metadata = f"direction_metadata{suffix}.json"
    #     vector = t.load(os.path.join(PROJECT_DIR, base_path, f_name_steer), map_location="cpu")
    #     with open(os.path.join(base_path, f_name_metadata)) as f:
    #         meta_data = json.load(f)

    #     layer_idx, pos = meta_data['layer'], meta_data['pos']

    #     with open(f"pipeline/runs/{model_path.split('/')[-1]}/select_direction/direction_evaluations_filtered.json", 'r') as f:
    #         filtered_scores = json.load(f)

    #     sort_fn = lambda a, b: t.sigmoid(t.tensor(a)) - t.sigmoid(t.tensor(b))
    #     sorted_scores = sorted(filtered_scores, key=lambda x: sort_fn(x['steering_harmless_score'], x['steering_harmful_score']), reverse=True)
    #     pos_test = sorted_scores[0]['position']
    #     layer_idx_test = sorted_scores[0]['layer']

    #     assert layer_idx == layer_idx_test and pos == pos_test
    
    coeff = -1 if harm_flag else 1

    vector = t.load(
        f"refusal_direction/pipeline/runs/{model_path.split('/')[-1]}/direction_layer{layer_idx}_pos{pos}.pt",
        map_location="cpu"
    )

    print(f"loaded vector with shape {vector.shape}, coefficient {coeff}, from layer {layer_idx}, eoi token {pos}")

    return vector, coeff, layer_idx

def _load_learned_vector(concept, model_path, concept_params, layer_idx, vector_base_path):
    if concept == 'refusal':
        harm_flag = concept_params['harm_flag']
        coeff_sign = -1 if harm_flag else 1
    else:
        raise ValueError(f"other concepts not supported yet, received {concept}")
    try:
        trained_vector_dir = f"trained_vectors/{model_path.split('/')[-1]}/{vector_base_path}"
        print(f'loading learned vector from {trained_vector_dir}')
        val_f_name = f"val.json"
        with open(os.path.join(trained_vector_dir, val_f_name), 'r') as f:
            val_data = json.load(f)
            epoch = val_data['best_epoch']
            best_norm = val_data['best_norm']
            print(f'using epoch {epoch} with norm {best_norm}')

        vector_f_name = f"direction_{epoch}.pt"
        vector = t.load(os.path.join(trained_vector_dir, vector_f_name))
        metadata_f_name = f"direction_{epoch}_metadata.json"
        with open(os.path.join(trained_vector_dir, metadata_f_name), 'r') as f:
            metadata = json.load(f)
            metadata_layer = metadata['layer']
            assert metadata_layer == layer_idx

        # use dim norm
        print(f'current norm: {vector.norm()}')
        vector = vector / vector.norm().item() * best_norm

        return vector, coeff_sign, layer_idx
    except:
        print(f'assuming is from vectors dir: {vector_base_path}')
        data = t.load(vector_base_path)
        vector = data['direction']
        layer = data['layer']
        
        return vector, coeff_sign, layer


def load_steering_vector(concept: str, model_path: str, concept_params: dict, pos: int, layer_idx: int, learned=False, check_best=False, vector_base_path=None, **kwargs):
    """
    Returns steering vector and layer idx. make sure steering vector is prehooked into the layer at layer_idx
    """
    if concept == "refusal":
        if not learned:
            vector, coeff, layer_idx = _load_refusal_dim_vector(concept, model_path, concept_params, pos, layer_idx, check_best)
        else:
            assert vector_base_path is not None
            vector, coeff, layer_idx = _load_learned_vector(concept, model_path, concept_params, layer_idx, vector_base_path)
        return vector, coeff, layer_idx

    else:
        raise ValueError(f"concept {concept} is not yet supported")
