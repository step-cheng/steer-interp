import torch as t
from nnsight import LanguageModel
from utils.data_utils import load_steering_vector, load_refusal_completions
from data_preprocess import format_input
from transformers import AutoTokenizer, AutoModelForCausalLM
from patching.loading_utils import load_submodules, Submodule, SubmoduleStash
from functools import partial
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import os
import random
from iic_node import IICNode
from dataset.load_dataset import load_dataset, load_dataset_split
from utils.gen_utils import load_model_and_tokenizer

TEST_DATASETS = ['jailbreakbench', 'alpaca']

def load_test_data(dataset_name, n_examples=100):
    random.seed(42)
    if dataset_name == 'alpaca':
        with open('pipeline/runs/gemma-2-2b-it/completions_layer15_pos-1/harmless_actadd_completions.json', 'r') as f:
            completions = json.load(f)
        data = [{
            'category': s['category'], 'instruction': s['prompt']
        } for s in completions]
    elif dataset_name in TEST_DATASETS:
        data = load_dataset(dataset_name, False)
    else:
        raise ValueError(f'unknown dataset {dataset_name}')
    return data

def _steer(resid_submod, steer_vec):
    resid_base = resid_submod.get_out_activation().clone()
    resid_steer = resid_base + steer_vec.to(resid_submod.device)
    resid_submod.set_out_activation(resid_steer)

def get_steer_mlp_submodules(steer_submodules: SubmoduleStash, steer_idx):
    mlp_submodules = []
    for i in range(steer_idx, len(steer_submodules.mlps)):
        mlp_submodules.append(steer_submodules.mlps[i])
    return mlp_submodules

def get_steer_attn_submodules(steer_submodules: SubmoduleStash, steer_idx):
    attn_submodules = []
    for i in range(steer_idx, len(steer_submodules.attns)):
        attn_submodules.append(steer_submodules.attns[i])
    return attn_submodules

def base_generate(model, prompt_inputs, steer_fn, max_new_tokens, return_logits:bool=False, **kwargs):
    all_logits_base = []
    decoded_tokens = []
    with t.no_grad(), model.generate(prompt_inputs, max_new_tokens=max_new_tokens, do_sample=False) as tracer:
        with tracer.iter[:]:
            logits = model.lm_head.output.save()
            tokens_greedy = logits[:,-1].argmax(dim=-1)
            decoded_tokens.append(tokens_greedy.save())
            all_logits_base.append(logits[:,-1])

    if return_logits:
        return t.stack(decoded_tokens, dim=-1).cpu(), t.stack(all_logits_base, dim=0).cpu()
    return t.stack(decoded_tokens, dim=-1).cpu()

def steer_generate(model, prompt_inputs, steer_fn, max_new_tokens, return_logits:bool=False, **kwargs):
    all_logits_steer = []
    decoded_tokens = []
    with t.no_grad(), model.generate(prompt_inputs, max_new_tokens=max_new_tokens, do_sample=False) as tracer:
        with tracer.iter[:]:
            steer_fn()
            logits = model.lm_head.output.save()
            tokens_greedy = logits[:,-1].argmax(dim=-1)
            decoded_tokens.append(tokens_greedy.save())
            all_logits_steer.append(logits[:,-1])

    if return_logits:
        return t.stack(decoded_tokens, dim=-1).cpu(), t.stack(all_logits_steer, dim=0).cpu()
    return t.stack(decoded_tokens, dim=-1).cpu()

def freeze_attn_weight_pass(model, prompt_inputs, steer_submodules, steer_layer_idx, steer_fn):
    attn_weights_base = {}
    attn_submodules = get_steer_attn_submodules(steer_submodules, steer_layer_idx)
    prompt_ids = prompt_inputs['input_ids']
    b, n = prompt_ids.shape
    with t.no_grad(), model.trace(prompt_inputs):
        for attn in attn_submodules:
            attn_out_tuple = attn.submodule.source.attention_interface_0.output
            attn_weights_base[attn] = attn_out_tuple[1].save()
        logits_base = model.lm_head.output.save()
    
    with t.no_grad(), model.trace(prompt_inputs):
        steer_fn()
        for attn in attn_submodules:
            attn_base = attn_weights_base[attn].to(attn.device)

            # reimplement attention
            values = attn.submodule.v_proj.output
            values = values.view(b, n, -1, attn.submodule.head_dim).transpose(1,2)
            values = t.repeat_interleave(values, dim=1, repeats=attn.submodule.num_key_value_groups)
            new_attn_out = t.matmul(attn_base, values)
            new_attn_out = new_attn_out.transpose(1,2).contiguous()
            new_attn_out = new_attn_out.reshape(b,n,-1).contiguous()
            attn.submodule.o_proj.input = new_attn_out
        logits_freeze = model.lm_head.output.save()

    return logits_freeze, logits_base, attn_weights_base

def freeze_attn_plus_iic_pass(model, prompt_inputs, steer_submodules, steer_layer_idx, steer_fn):
    attn_weights_base = {}
    attn_submodules = get_steer_attn_submodules(steer_submodules, steer_layer_idx)
    prompt_ids = prompt_inputs['input_ids']
    b, n = prompt_ids.shape
    with t.no_grad(), model.trace(prompt_inputs):
        for attn in attn_submodules:
            attn_out_tuple = attn.submodule.source.attention_interface_0.output
            attn_weights_base[attn] = attn_out_tuple[1].save()
        logits_base = model.lm_head.output.save()
    
    with t.no_grad(), model.trace(prompt_inputs):
        steer_fn()
        for attn in attn_submodules:
            attn_base = attn_weights_base[attn].to(attn.device)

            # reimplement attention
            values = attn.submodule.v_proj.output
            values = values.view(b, n, -1, attn.submodule.head_dim).transpose(1,2)
            values = t.repeat_interleave(values, dim=1, repeats=attn.submodule.num_key_value_groups)
            new_attn_out = t.matmul(attn_base, values)
            new_attn_out = new_attn_out.transpose(1,2).contiguous()
            new_attn_out = new_attn_out.reshape(b,n,-1).contiguous()
            attn.submodule.o_proj.input = new_attn_out
        logits_freeze = model.lm_head.output.save()

    return logits_freeze, logits_base, attn_weights_base

def generate_no_iic(model, prompt_inputs, submodules, steer_fn, steer_layer_idx, max_new_tokens, steer_vec, freeze_pass_fn=None):
    """
    Delete steer vec from all attention value inputs.
    """
    attn_submodules = get_steer_attn_submodules(submodules, steer_layer_idx)
    decoded_tokens_freeze = []
    
    with t.no_grad(), model.generate(prompt_inputs, max_new_tokens=max_new_tokens, do_sample=False) as tracer:
        with tracer.iter[:]:
            steer_fn()
            for attn in attn_submodules:
                iic_node = IICNode(model, attn.layer_idx)
                steer_vec = steer_vec.to(attn.device)
                
                scales = iic_node.get_pre_rms_norm_stat(attn.pre_ln.input.detach()) # b s 1
                to_subtract = iic_node.pre_ln_norm_v(steer_vec, scales)
                
                value_in = attn.submodule.v_proj.input
                new_value_in = value_in - to_subtract
                attn.submodule.v_proj.input = new_value_in
            logits = model.lm_head.output.save()
            tokens_greedy = logits[:,-1].argmax(dim=-1)
            decoded_tokens_freeze.append(tokens_greedy.save())

    return t.stack(decoded_tokens_freeze, dim=-1).cpu()

def generate_no_mlp_direct(model, prompt_inputs, submodules, steer_fn, steer_layer_idx, max_new_tokens, steer_vec, freeze_pass_fn=None):
    """
    Delete steer vec from all mlp value inputs
    """
    mlp_submodules = get_steer_mlp_submodules(submodules, steer_layer_idx)
    decoded_tokens_freeze = []

    def _get_pre_rms_norm_stat(x, ln_eps):
        return t.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + ln_eps)
    
    def _pre_ln_norm_v(raw_in_v, ln_weight, ln_stats=1):
        in_v = raw_in_v.float() * ln_stats
        if model.config._name_or_path.startswith('google/gemma-2-'):
            in_v = in_v * (1.0 + ln_weight.float())
        if model.config._name_or_path.startswith('meta-llama/Llama-3.2-'):
            in_v = in_v * ln_weight.float()
        return in_v.type_as(raw_in_v)
    
    with t.no_grad(), model.generate(prompt_inputs, max_new_tokens=max_new_tokens, do_sample=False) as tracer:
        with tracer.iter[:]:
            steer_fn()
            for mlp in mlp_submodules:
                pre_ln = mlp.pre_ln
                ln_eps = pre_ln.eps if hasattr(pre_ln, 'eps') else pre_ln.variance_epsilon
                ln_weight = pre_ln.weight

                scales = _get_pre_rms_norm_stat(pre_ln.input.detach(), ln_eps) # b s 1
                steer_vec = steer_vec.to(mlp.device)
                to_subtract = _pre_ln_norm_v(steer_vec, ln_weight, ln_stats=scales)

                mlp_inp = mlp.get_in_activation().detach()
                new_mlp_inp = mlp_inp - to_subtract
                mlp.set_in_activation(new_mlp_inp)
            logits = model.lm_head.output.save()
            tokens_greedy = logits[:,-1].argmax(dim=-1)
            decoded_tokens_freeze.append(tokens_greedy.save())
                
    return t.stack(decoded_tokens_freeze, dim=-1).cpu()

def freeze_attn_value_pass(model, prompt_inputs, steer_submodules, steer_layer_idx, steer_fn):
    attn_vals_base = {}
    attn_submodules = get_steer_attn_submodules(steer_submodules, steer_layer_idx)
    with t.no_grad(), model.trace(prompt_inputs):
        for attn in attn_submodules:
            v_out = attn.submodule.v_proj.output
            attn_vals_base[attn] = v_out.save()
        logits_base = model.lm_head.output.save()

    with t.no_grad(), model.trace(prompt_inputs):
        steer_fn()
        for attn in attn_submodules:
            attn_vals = attn_vals_base[attn]

            attn.submodule.v_proj.output = attn_vals.to(attn.device)
        logits_freeze = model.lm_head.output.save()

    return logits_freeze, logits_base, attn_vals_base

def freeze_mlp_pass(model, prompt_inputs, steer_submodules, steer_layer_idx, steer_fn):
    mlps_base = {}
    mlp_submodules = get_steer_mlp_submodules(steer_submodules, steer_layer_idx)
    with t.no_grad(), model.trace(prompt_inputs):
        for mlp in mlp_submodules:
            mlp_out = mlp.get_out_activation()
            mlps_base[mlp] = mlp_out.save()  
        logits_base = model.lm_head.output.save()

    with t.no_grad(), model.trace(prompt_inputs):
        steer_fn()
        for mlp in mlp_submodules:
            mlp_vals = mlps_base[mlp]

            mlp.set_out_activation(mlp_vals)
        logits_freeze = model.lm_head.output.save()

    return logits_freeze, logits_base, mlps_base

# def follow_steer(model, prompt_inputs, steer_submodules, steer_fn, max_new_tokens, steer_vec):
#     steer_tokens, all_logits_steer = steer_generate(model, prompt_inputs, steer_fn, max_new_tokens)
#     print(f'all logits steer shape: {all_logits_steer.shape}')

#     freeze_tokens = t.zeros_like(steer_tokens, dtype=t.int)
#     base_tokens = t.zeros_like(steer_tokens, dtype=t.int)
#     steer_ms = t.zeros_like(steer_tokens, dtype=t.int)
#     freeze_ms = t.zeros_like(steer_tokens, dtype=t.int)
#     base_ms = t.zeros_like(steer_tokens, dtype=t.int)
#     for i in range(max_new_tokens):
#         logits_freeze = freeze_pass(model, prompt_inputs, steer_submodules, steer_fn)
        
#         base_y = t.argmax(logits_base[0,-1], dim=-1).flatten()
#         base_tokens[i] = base_y
#         base_logit_diff = logits_base[0, -1, steer_tokens[i]] - logits_base[0, -1, base_y]
#         base_ms[i] = base_logit_diff.cpu()

#         freeze_y = t.argmax(logits_freeze[0, -1], dim=-1).flatten()
#         freeze_tokens[i] = freeze_y
#         freeze_logit_diff = logits_freeze[0, -1, steer_tokens[i]] - logits_freeze[0, -1, base_y]
#         freeze_ms[i] = freeze_logit_diff.cpu()

#         steer_logit_diff = all_logits_steer[i, steer_tokens[i]] - all_logits_steer[i, base_y] 
#         steer_ms[i] = steer_logit_diff.cpu()

#         prompt_inputs = t.cat((prompt_inputs, steer_tokens[i].unsqueeze(0).unsqueeze(0)), dim=-1)

#     return steer_tokens, freeze_tokens, base_tokens, steer_ms, freeze_ms, base_ms

def generate_freeze(model, prompt_inputs, submodules, steer_fn, steer_layer_idx, max_new_tokens, freeze_pass_fn):
    prompt_ids = prompt_inputs['input_ids']
    attn_mask = prompt_inputs['attention_mask']
    bsz = prompt_ids.shape[0]
    freeze_tokens = t.zeros((bsz,max_new_tokens), dtype=t.int)

    eos_id = tokenizer.eos_token_id
    active_mask = t.ones(bsz, dtype=t.bool)

    for i in range(max_new_tokens):
        logits_freeze, logits_base, _ = freeze_pass_fn(model, prompt_inputs, submodules, steer_layer_idx, steer_fn)
        freeze_y = t.argmax(logits_freeze[:, -1], dim=-1).flatten()
        freeze_tokens[:, i] = freeze_y

        prompt_ids = t.cat((prompt_ids, freeze_y.unsqueeze(1).cpu()), dim=-1)
        attn_mask = t.cat((attn_mask, t.ones((bsz, 1)).cpu()), dim=-1)
        prompt_inputs = {
            'input_ids': prompt_ids,
            'attention_mask': attn_mask,
        }

    return freeze_tokens

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--dataset', type=str, choices=TEST_DATASETS)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--freeze_type', type=str, choices=['attn_weights', 'attn_vals', 'mlps', 'iic', 'none', 'mlp_direct', 'test'])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_tokens', type=int, default=512)
    args = parser.parse_args()

    # load data
    dataset_name = args.dataset
    dataset = load_test_data(dataset_name)
    n_examples = len(dataset)

    # load model
    model_path = args.model_path
    model = LanguageModel(model_path, dispatch=True, dtype=t.bfloat16, device_map='auto')
    if args.freeze_type == "attn_weights":
        model.set_attn_implementation("eager")
    print(f'model training: {model.training}')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    # load vector
    harm_flag = False if dataset_name == 'alpaca' else True
    layer, pos = {
        'google/gemma-2-2b-it': (15, -1),
        'google/gemma-2-9b-it': (23, -1),
        'meta-llama/Llama-3.2-3B-Instruct': (12, -4),
        'Qwen/Qwen3-8B': (20, -8),
    }[model_path]
    vector, coeff, steer_layer_idx = load_steering_vector("refusal", model_path, {'harm_flag': harm_flag}, pos, layer)
    print(f'loaded vector {vector.shape} with coeff {coeff} at layer {steer_layer_idx}')
    steer_vec = vector.to(t.bfloat16) * coeff

    # load submodules
    submodules = load_submodules(model, True)
    resid_submod = submodules.resids[steer_layer_idx-1]
    
    steer_fn = partial(_steer, resid_submod=resid_submod, steer_vec=steer_vec)
    generate_fn, freeze_pass_fn = {
        'attn_weights': (generate_freeze, freeze_attn_weight_pass),
        'attn_vals': (generate_freeze, freeze_attn_value_pass),
        'mlps': (generate_freeze, freeze_mlp_pass),
        'iic': (partial(generate_no_iic, steer_vec=steer_vec), None),
        'none': (partial(steer_generate, steer_vec=steer_vec), None),
        'mlp_direct': (partial(generate_no_mlp_direct, steer_vec=steer_vec), None),
        'test': (partial(base_generate, steer_vec=steer_vec), None)
    }[args.freeze_type]
    max_new_tokens= args.max_tokens

    steer_flag = True

    save_dir = f"freeze_generations/{model_path.split('/')[-1]}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{args.freeze_type}_{dataset_name}.jsonl"
    print(f'save path: {save_path}')
    start_idx = args.start_idx
    batch_size = args.batch_size

    for ex in tqdm(range(start_idx, n_examples, batch_size)):
        prompts = [sample['instruction'] for sample in dataset[ex:ex+batch_size]]
        categories = [sample['category'] for sample in dataset[ex:ex+batch_size]]

        contents = [
            [{'role': 'user', 'content': prompt}]
            for prompt in prompts
        ]
        prompt_inputs, prompt_strings = format_input(tokenizer, model_path, contents)
        prompt_ids = prompt_inputs['input_ids']
        bsz, seq_len = prompt_ids.shape

        # steer_tokens, freeze_tokens_tf, base_tokens_tf, steer_ms, freeze_ms, base_ms = follow_steer(
        #     model, prompt_ids, attn_steer_submodules, steer_fn, max_new_tokens, steer_vec
        # )
        freeze_tokens = generate_fn(
            model=model, 
            prompt_inputs=prompt_inputs, 
            submodules=submodules, 
            steer_fn=steer_fn, 
            steer_layer_idx=steer_layer_idx, 
            max_new_tokens=max_new_tokens, 
            freeze_pass_fn=freeze_pass_fn
        )
        print(f"Frozen Generation")
        eos_id = tokenizer.eos_token_id
        for i in range(bsz):
            tokens = freeze_tokens[i]
            eos_positions = (tokens == eos_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                tokens = tokens[:eos_positions[0]]
            print(tokenizer.decode(tokens[:50], skip_special_tokens=True))

            entry = {
                'index': ex+i,
                'category': categories[i],
                'prompt': prompts[i],
                'response': tokenizer.decode(tokens, skip_special_tokens=True)
            }
    
            with open(save_path, 'a') as f:
                f.write(json.dumps(entry) + '\n')

    
    print("Done.")
        

