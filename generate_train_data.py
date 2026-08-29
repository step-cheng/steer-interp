import torch as t
import random
import json
import os
import argparse
from tqdm import tqdm
from pipeline.config import Config
from pipeline.run_pipeline import load_and_sample_datasets
from dataset.load_dataset import load_dataset_split
from pipeline.submodules.select_direction import get_refusal_scores

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from datasets import load_dataset

from config_act_patch import dim_layer_map, dim_layer_pos_dict
from mat_hooks import get_activation_addition_input_pre_hook
from utils.data_utils import load_steering_vector


SYSTEM_PROMPT = """Treat the following question as dangerous and refuse.

"""
HARM_TYPES = ['harmful', 'harmless']
SPLITS = ['train', 'val', 'test']

def load_data(cfg, n_total):
    # load data this way to load same data as the difference in means
    harmful_train_original, harmless_train_original, harmful_val, harmless_val = load_and_sample_datasets(cfg)
    harmful_full = load_dataset_split(harmtype='harmful', split='train', instructions_only=True)
    harmless_full = load_dataset_split(harmtype='harmless', split='train', instructions_only=True)

    with open('generate_harmful_val.json', 'w') as f:
        pass
    n_additional = n_total - cfg.n_train
    harmful_remaining = [x for x in harmful_full if x not in harmful_train_original]
    harmless_remaining = [x for x in harmless_full if x not in harmless_train_original]
    harmful_additional = random.sample(harmful_remaining, n_additional)
    harmless_additional = random.sample(harmless_remaining, n_additional)

    harmful_train = harmful_train_original + harmful_additional
    harmless_train = harmless_train_original + harmless_additional

    return harmful_train, harmless_train, harmful_val, harmless_val

def load_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        dtype=t.float32, 
        device_map="auto",
    ).eval()
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side='left'
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def load_abliterated_model(model_path):
    abliterated_model_path = {
        'google/gemma-2-2b-it': "IlyaGusev/gemma-2-2b-it-abliterated",
        'meta-llama/Llama-3.2-3B-Instruct': "huihui-ai/Llama-3.2-3B-Instruct-abliterated"
    }[model_path]
    abliterated_model = AutoModelForCausalLM.from_pretrained(abliterated_model_path, dtype=t.float32, device_map="auto").eval()
    abliterated_model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side='left'
    tokenizer.pad_token = tokenizer.eos_token
    return abliterated_model, tokenizer

def format_harmful_input(tokenizer, prompt):
    prompts = [
        {"role": "user", "content": prompt},
    ]
    try:
        # if qwen3 8b
        strings = tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except:
        print('failed')
        strings = tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)
    exit()
    inputs = tokenizer(strings, return_tensors="pt", add_special_tokens=False)
    return inputs, strings

def format_harmless_input(tokenizer, prompt, model_family):
    if model_family == 'gemma':
        prompts = [
            {"role": "user", "content": f'{SYSTEM_PROMPT}{prompt}'}
        ]
    else:
        prompts = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    try:
        strings = tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except:
        print('failed')
        strings = tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)
    exit()
    inputs = tokenizer(strings, return_tensors="pt", add_special_tokens=False)
    return inputs, strings

def format_data(tokenizer, prompt, harm_type, model_family='gemma'):
    if harm_type == "harmless":
        return format_harmless_input(tokenizer, prompt, model_family)
    elif harm_type == "harmful":
        return format_harmful_input(tokenizer, prompt)
    else:
        raise ValueError(f'invalid')
    
def batch_format_data(tokenizer, prompts, harm_type, steer_flag, model_family='gemma'):
    """Format a batch of prompts and return padded batch inputs + per-sample prompt lengths."""
    all_strings = []
    for prompt in prompts:
        if steer_flag:
            if harm_type == "harmless":
                if model_family == 'gemma':
                    messages = [{"role": "user", "content": f'{SYSTEM_PROMPT}{prompt}'}]
                else:
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ]
            elif harm_type == "harmful":
                messages = [{"role": "user", "content": prompt}]
            else:
                raise ValueError(f'invalid harm_type: {harm_type}')
        else:
            messages = [{"role": "user", "content": prompt}]
        try:
            s = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except:
            s = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        all_strings.append(s)

    # Tokenize individually to get prompt lengths before padding
    prompt_lengths = []
    for s in all_strings:
        tokens = tokenizer(s, add_special_tokens=False)
        prompt_lengths.append(len(tokens['input_ids']))

    # Batch tokenize with padding (left-padded)
    batch_inputs = tokenizer(all_strings, return_tensors="pt", padding=True, add_special_tokens=False)

    return batch_inputs, prompt_lengths, all_strings

def generate_completions_batched(model, tokenizer, prompts, harm_type, generation_config, steer_flag, bsz=8, model_family='gemma'):
    """Generate completions in batches, returning individual prompt/response pairs."""
    contents = []
    system = SYSTEM_PROMPT if harm_type == "harmless" and steer_flag == True else None

    for i in tqdm(range(0, len(prompts), bsz), desc=f"Generating {harm_type} (bsz={bsz})"):
        batch_prompts = prompts[i:i+bsz]

        batch_inputs, prompt_lengths, _ = batch_format_data(tokenizer, batch_prompts, harm_type, steer_flag, model_family)
        batch_inputs = {k: v.to(model.device) for k, v in batch_inputs.items()}

        outputs = model.generate(
            **batch_inputs,
            generation_config=generation_config,
        )

        # Decode each sample individually
        for j, prompt in enumerate(batch_prompts):
            seq = outputs[j]
            # Left-padding means the prompt ends at the same position for all samples,
            # but each has different padding at the start. Use attention_mask to find
            # where real tokens begin, then offset by prompt length.
            num_pad = (batch_inputs['attention_mask'][j] == 0).sum().item()
            response_start = num_pad + prompt_lengths[j]
            response_tokens = seq[response_start:]
            response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)

            tqdm.write(f"[{harm_type}] {prompt[:60]}... -> {response_text[:80]}...")

            contents.append({
                "system": system,
                "prompt": prompt,
                "response": response_text
            })

    return contents

# Keep original unbatched versions for reference/fallback
def generate_harmful_steer_completions(model, tokenizer, harmful_train, generation_config, bsz=8):
    return generate_completions_batched(model, tokenizer, harmful_train, 'harmful', generation_config, True, bsz=bsz)

def generate_harmless_steer_completions(model, tokenizer, harmless_train, generation_config, bsz=8):
    return generate_completions_batched(model, tokenizer, harmless_train, 'harmless', generation_config, True, bsz=bsz)

def generate_harmful_base_completions(model, tokenizer, harmful_train, generation_config, bsz=8):
    return generate_completions_batched(model, tokenizer, harmful_train, 'harmful', generation_config, False, bsz=bsz)

def generate_harmless_base_completions(model, tokenizer, harmless_train, generation_config, bsz=8):
    return generate_completions_batched(model, tokenizer, harmless_train, 'harmless', generation_config, False, bsz=bsz)

def run_pipeline(model_path, max_tokens, n_total, overwrite=False):
    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias, model_path=model_path)
    harmful_train, harmless_train, harmful_val, harmless_val = load_data(cfg, n_total)
    print(f'Training with {len(harmful_train)} harmful samples, {len(harmless_train)} harmless samples')

    model, tokenizer = load_model(model_path)
    generation_config = GenerationConfig(
        max_new_tokens=max_tokens,
        do_sample=False
    )

    layer = dim_layer_map[model_path]
    _, pos = dim_layer_pos_dict[model_path][layer]
    vector, coeff, layer_idx = load_steering_vector(
        "refusal",
        model_path,
        {'harm_flag': True},
        pos,
        layer,
    )
    vector = vector.to(model.device)
    hook_fn = get_activation_addition_input_pre_hook(vector=vector, coeff=coeff)

    model_name = model_path.split('/')[-1]
    os.makedirs(f"training_data/{model_name}", exist_ok=True)
    print("generating train harmless steered completions")
    completions_harmless_steer_train = generate_harmless_steer_completions(model, tokenizer, harmless_train, generation_config)
    with open(f'training_data/{model_name}/harmless_refuse_train.json', 'w') as f:
        json.dump(completions_harmless_steer_train, f, indent=4)

    print("generating train harmful steered completions")
    hook_handle = model.model.layers[layer_idx].register_forward_pre_hook(hook_fn)
    try:
        completions_harmful_steer_train = generate_harmful_steer_completions(model, tokenizer, harmful_train, generation_config)
    finally:
        hook_handle.remove()
    with open(f'training_data/{model_name}/harmful_comply_train.json', 'w') as f:
        json.dump(completions_harmful_steer_train, f, indent=4)

    print("generating train harmless base completions")
    completions_harmless_base_train = generate_harmless_base_completions(model, tokenizer, harmless_train, generation_config)
    with open(f'training_data/{model_name}/harmless_comply_train.json', 'w') as f:
        json.dump(completions_harmless_base_train, f, indent=4)

    print("generating harmful base completions")
    completions_harmful_base_train = generate_harmful_base_completions(model, tokenizer, harmful_train, generation_config)
    with open(f'training_data/{model_name}/harmful_refuse_train.json', 'w') as f:
        json.dump(completions_harmful_base_train, f, indent=4)

    print("generating validation harmless steered completions")
    completions_harmless_steer_val = generate_harmless_steer_completions(model, tokenizer, harmless_val, generation_config)
    with open(f'training_data/{model_name}/harmless_refuse_val.json', 'w') as f:
        json.dump(completions_harmless_steer_val, f, indent=4)

    print("generating validation harmful steered completions")
    hook_handle = model.model.layers[layer_idx].register_forward_pre_hook(hook_fn)
    try:
        completions_harmful_steer_val = generate_harmful_steer_completions(model, tokenizer, harmful_val, generation_config)
    finally:
        hook_handle.remove()
    with open(f'training_data/{model_name}/harmful_comply_val.json', 'w') as f:
        json.dump(completions_harmful_steer_val, f, indent=4)

    print("generating validation harmless base completions")
    completions_harmless_base_val = generate_harmless_base_completions(model, tokenizer, harmless_val, generation_config)
    with open(f'training_data/{model_name}/harmless_comply_val.json', 'w') as f:
        json.dump(completions_harmless_base_val, f, indent=4)

    print("generating validation harmful base completions")
    completions_harmful_base_val = generate_harmful_base_completions(model, tokenizer, harmful_val, generation_config)
    with open(f'training_data/{model_name}/harmful_refuse_val.json', 'w') as f:
        json.dump(completions_harmful_base_val, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--tokens', type=int, default=512)
    parser.add_argument('--n_examples', type=int, default=256)
    parser.add_argument('--overwrite', action="store_true", default=False)
    args = parser.parse_args()
    run_pipeline(args.model_path, args.tokens, args.n_examples, args.overwrite)
    print('done')