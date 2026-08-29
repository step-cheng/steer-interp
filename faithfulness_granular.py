import torch as t
import numpy as np
from nnsight import LanguageModel
from patching.loading_utils import Submodule
from collections import defaultdict
from patching.loading_utils import load_submodules, Submodule, SubmoduleStash
import os
import math
import json
from utils.data_utils import load_steering_vector
import pandas as pd
from tqdm import tqdm
from patching.attribution import get_steer_submodules
from transformers import AutoTokenizer
from einops import einsum
import torch.nn.functional as F
from config_act_patch import Config, dim_layer_pos_dict, dim_layer_map
from copy import deepcopy
from utils.data_utils import load_refusal_completions
from data_preprocess import format_input

def prep_data(model_path, concept_params, tokenizer, pos, layer, learned, learn_path=None):
    all_completions = load_refusal_completions(model_path, True, concept_params, learned, pos, layer, learn_path)
    examples = []

    for prompt_contents in all_completions:
        prompt_inputs, prompt_strings = format_input(tokenizer, model_path, [[prompt_contents[0]]])

        prompt_len = len(prompt_inputs['input_ids'][0])

        convo_inputs, convo_strings = format_input(tokenizer, model_path, [prompt_contents])

        examples.append({
            'convo': convo_strings,
            'prompt': prompt_strings,
            'input_ids': convo_inputs['input_ids'],
            'prompt_length': prompt_len,
        })
    return examples

# Metric Functions
def metric_logit_fn(logits_steer, logits_base, logits_circuit, steer_ys):
    base_ys = t.argmax(logits_base[:, -1, :], dim=-1) # b s d
    steer_ys = t.argmax(logits_base[:, -1, :], dim=-1) # b s d
    n_batch = len(logits_steer)
    mismatches = base_ys != steer_ys
    m_base = (logits_base[t.arange(n_batch), -1, steer_ys] - logits_base[t.arange(n_batch), -1, base_ys]).mean(dim=0).item()
    m_steer = (logits_steer[t.arange(n_batch), -1, steer_ys] - logits_steer[t.arange(n_batch), -1, base_ys]).mean(dim=0).item()
    m_circuit = (logits_circuit[t.arange(n_batch), -1, steer_ys] - logits_circuit[t.arange(n_batch), -1, base_ys]).mean(dim=0).item()
    return (m_circuit - m_base) / (m_steer - m_base)

def metric_match_fn(logits_steer, logits_base, logits_circuit, prompt_len):
    base_ys = t.argmax(logits_base[:, prompt_len-1:-1, :], dim=-1)
    circuit_ys = t.argmax(logits_circuit[:, prompt_len-1:-1, :], dim=-1)
    new_steer_ys = t.argmax(logits_steer[:, prompt_len-1:-1, :], dim=-1)

    base_matches = base_ys == new_steer_ys
    circuit_matches = circuit_ys == new_steer_ys

    return base_matches, circuit_matches

def metric_kl1_fn(logits_steer, logits_base, logits_circuit, prompt_len):
    probs_steer = F.softmax(logits_steer[:,prompt_len-1:-1], dim=-1)
    log_probs_base = F.log_softmax(logits_base[:,prompt_len-1:-1], dim=-1)
    log_probs_circuit = F.log_softmax(logits_circuit[:,prompt_len-1:-1], dim=-1)

    m_base = F.kl_div(log_probs_base, probs_steer, reduction='none').sum(dim=-1)
    m_steer = 0
    m_circuit = F.kl_div(log_probs_circuit, probs_steer, reduction='none').sum(dim=-1)
    return 1 - m_circuit / m_base

def metric_kl2_fn(logits_steer, logits_base, logits_circuit, prompt_len):
    logits_base_gen = logits_base[:, prompt_len-1:-1]
    logits_steer_gen = logits_steer[:, prompt_len-1:-1]
    logits_circuit_gen = logits_circuit[:, prompt_len-1:-1]

    probs_base = F.softmax(logits_base_gen, dim=-1)
    probs_steer = F.softmax(logits_steer_gen, dim=-1)
    log_probs_base = F.log_softmax(logits_base_gen, dim=-1)
    log_probs_steer = F.log_softmax(logits_steer_gen, dim=-1)
    log_probs_circuit = F.log_softmax(logits_circuit_gen, dim=-1)
    
    m_base = 0 - F.kl_div(log_probs_base, probs_steer, reduction='none').sum(dim=-1)
    m_steer = F.kl_div(log_probs_steer, probs_base, reduction='none').sum(dim=-1) - 0
    m_circuit = F.kl_div(log_probs_circuit, probs_base, reduction='none').sum(dim=-1) - \
                F.kl_div(log_probs_circuit, probs_steer, reduction='none').sum(dim=-1)
    return (m_circuit - m_base) / (m_steer - m_base + 1e-7)

def metric_logprob_fn(logits_steer, logits_base, logits_circuit, prompt_len):
    log_probs_base = F.log_softmax(logits_base[:, prompt_len-1:-1].to(t.float64), dim=-1)
    log_probs_steer = F.log_softmax(logits_steer[:, prompt_len-1:-1].to(t.float64), dim=-1)
    log_probs_circuit = F.log_softmax(logits_circuit[:, prompt_len-1:-1].to(t.float64), dim=-1)

    greedy_steer_inds = t.argmax(log_probs_steer, dim=-1)
    greedy_base_inds = t.argmax(log_probs_base, dim=-1)

    steer_gen_from_steer = t.gather(log_probs_steer, dim=-1, index=greedy_steer_inds.unsqueeze(-1)).squeeze(-1)
    steer_gen_from_base = t.gather(log_probs_base, dim=-1, index=greedy_steer_inds.unsqueeze(-1)).squeeze(-1)
    steer_gen_from_circuit = t.gather(log_probs_circuit, dim=-1, index=greedy_steer_inds.unsqueeze(-1)).squeeze(-1)

    base_gen_from_steer = t.gather(log_probs_steer, dim=-1, index=greedy_base_inds.unsqueeze(-1)).squeeze(-1)
    base_gen_from_base = t.gather(log_probs_base, dim=-1, index=greedy_base_inds.unsqueeze(-1)).squeeze(-1)
    base_gen_from_circuit = t.gather(log_probs_circuit, dim=-1, index=greedy_base_inds.unsqueeze(-1)).squeeze(-1)

    steer_gen_steer_prob = steer_gen_from_steer.sum(dim=-1)
    steer_gen_base_prob = steer_gen_from_base.sum(dim=-1)
    steer_gen_circuit_prob = steer_gen_from_circuit.sum(dim=-1)
    
    base_gen_steer_prob = base_gen_from_steer.sum(dim=-1)
    base_gen_base_prob = base_gen_from_base.sum(dim=-1)
    base_gen_circuit_prob = base_gen_from_circuit.sum(dim=-1)

    diff_steer = steer_gen_steer_prob - base_gen_steer_prob
    diff_base = steer_gen_base_prob - base_gen_base_prob
    diff_circuit = steer_gen_circuit_prob - base_gen_circuit_prob
    
    return (diff_circuit - diff_base) / (diff_steer - diff_base)

def gemma_rms_norm_manual(ln_mod, x):
    eps = ln_mod.eps
    weight = ln_mod.weight
    rms_weight = t.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    output = x.float() * rms_weight
    output = output * (1.0 + weight.float())
    return output.type_as(x)

def llama_rms_norm_manual(ln_mod, x):
    eps = ln_mod.variance_epsilon
    weight = ln_mod.weight
    rms_weight = t.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    output = x.float() * rms_weight
    output = output.type_as(x) * weight
    return output

def qwen3_rms_norm_manual(ln_mod, x):
    eps = ln_mod.variance_epsilon
    weight = ln_mod.weight
    rms_weight = t.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    output = x.float() * rms_weight
    output = output.type_as(x) * weight
    return output

def get_rms_manual_fn(model):
    fn_dict = {
        'google/gemma-2-2b-it': gemma_rms_norm_manual,
        'meta-llama/Llama-3.2-3B-Instruct': llama_rms_norm_manual,
        'Qwen/Qwen3-8B': qwen3_rms_norm_manual
    }
    return fn_dict[model.config._name_or_path]

def forward_with_circuit(model, submodules: list[Submodule], steer_submod: Submodule, steer_vec, inputs, input_masks, node_names, base_acts):
    num_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    n_upstream_nodes = len(node_names)-1
    rms_norm_fn = get_rms_manual_fn(model)
    with t.no_grad(), model.trace(inputs):
        base_resid = steer_submod.get_out_activation().clone()
        steer_submod.set_out_activation(base_resid + steer_vec.to(steer_submod.device))
        act_deltas = t.zeros(
            (*inputs.shape, n_upstream_nodes, model.config.hidden_size),
            device=model.device, dtype=t.bfloat16
        )
        for submodule in submodules:
            # print(f'in submodule: {submodule.name}')
            if submodule.is_resid:
                # this is the steer submodule, just skip since no edges going to this
                assert submodule.name not in input_masks, f"{submodule.name}, {input_masks[submodule.name]}"
                act_deltas[:,:,0,:] = -steer_vec.to(submodule.device) # corrupt - clean. think of steered as clean
                act_deltas[:,:,0,:] = base_acts[submodule.name] - submodule.get_out_activation().clone() # corrupt - clean. think of steered as clean

            elif submodule.is_attn:
                ln_submod = submodule.pre_ln
                pre_ln_act = ln_submod.input
                attn = submodule.submodule
                head_dim = attn.head_dim

                # Build Q per attention head
                Q_parts = []
                for h in range(num_heads):
                    q_key = f"{submodule.name}_h{h}_q"
                    q_mask = (1 - input_masks[q_key]['mask']).to(submodule.device)
                    num_upstream = input_masks[q_key]['slice']
                    q_deltas = einsum(act_deltas[:,:,:num_upstream], q_mask[:num_upstream],
                                    "b s n_upstream d, n_upstream -> b s d")
                    q_input = rms_norm_fn(ln_submod, pre_ln_act + q_deltas)
                    W_q_h = attn.q_proj.weight[h*head_dim:(h+1)*head_dim, :]
                    Q_h = t.matmul(q_input, W_q_h.T)
                    if attn.q_proj.bias is not None:
                        Q_h = Q_h + attn.q_proj.bias[h*head_dim:(h+1)*head_dim]
                    Q_parts.append(Q_h)
                Q = t.cat(Q_parts, dim=-1)

                # Build K, V per KV group
                K_parts = []
                for g in range(num_kv_heads):
                    k_key = f"{submodule.name}_h{g}_k"
                    k_mask = (1 - input_masks[k_key]['mask']).to(submodule.device)
                    num_upstream = input_masks[k_key]['slice']
                    k_deltas = einsum(act_deltas[:,:,:num_upstream], k_mask[:num_upstream],
                                    "b s n_upstream d, n_upstream -> b s d")
                    k_input = rms_norm_fn(ln_submod, pre_ln_act + k_deltas)
                    W_k_g = attn.k_proj.weight[g*head_dim:(g+1)*head_dim, :]
                    K_g = t.matmul(k_input, W_k_g.T)
                    if attn.k_proj.bias is not None:
                        K_g = K_g + attn.k_proj.bias[g*head_dim:(g+1)*head_dim]
                    K_parts.append(K_g)
                K = t.cat(K_parts, dim=-1)

                V_parts = []
                for g in range(num_kv_heads):
                    v_key = f"{submodule.name}_h{g}_v"
                    v_mask = (1 - input_masks[v_key]['mask']).to(submodule.device)
                    num_upstream = input_masks[v_key]['slice']
                    v_deltas = einsum(act_deltas[:,:,:num_upstream], v_mask[:num_upstream],
                                    "b s n_upstream d, n_upstream -> b s d")
                    v_input = rms_norm_fn(ln_submod, pre_ln_act + v_deltas)
                    W_v_g = attn.v_proj.weight[g*head_dim:(g+1)*head_dim, :]
                    V_g = t.matmul(v_input, W_v_g.T)
                    if attn.v_proj.bias is not None:
                        V_g = V_g + attn.v_proj.bias[g*head_dim:(g+1)*head_dim]
                    V_parts.append(V_g)
                V = t.cat(V_parts, dim=-1)

                # Set projection outputs
                attn.q_proj.output = Q
                attn.k_proj.output = K
                attn.v_proj.output = V

                head_acts = submodule.get_per_head_outputs().clone()
                head_acts = submodule.handle_post_attn_mlp_norm(model, head_acts)
                for h in range(num_heads):
                    key = f"{submodule.name}_h{h}"
                    idx = node_names.index(key)
                    act_deltas[:, :, idx, :] = base_acts[key].to(submodule.device) - head_acts[:,:,h,:]
            else:
                ln_submod = submodule.pre_ln
                pre_ln_act = ln_submod.input

                mask = (1-input_masks[f"{submodule.name}"]['mask']).to(submodule.device)
                num_upstream = input_masks[f"{submodule.name}"]['slice']
                deltas = einsum(act_deltas[:,:,:num_upstream], mask[:num_upstream], 
                                "b s n_upstream d, n_upstream -> b s d")
                # print((pre_ln_act + deltas)[0,-1], base_ln_ins[submodule.name][0,-1])
                new_act = rms_norm_fn(ln_submod, pre_ln_act + deltas)
                submodule.set_in_activation(new_act)

                if submodule.is_mlp:
                    out_act = submodule.get_out_activation()
                    idx = node_names.index(submodule.name)
                    # print(f'setting idx {idx} with delta {(base_acts[submodule.name].to(submodule.device) - out_act)[0,-1]}')
                    act_deltas[:, :, idx, :] = base_acts[submodule.name].to(submodule.device) - out_act
        
        logits_circuit = model.lm_head.output.save()
    
    return logits_circuit

def forward_base(model, submodules: list[Submodule], inputs):
    num_heads = model.config.num_attention_heads
    hidden_states_base = {}
    ln_ins_base = {}
    with t.no_grad(), model.trace(inputs):
        for submodule in submodules:
            if not submodule.is_resid:
                ln_mod = submodule.pre_ln
                ins = ln_mod.input.detach()
                ln_ins_base[submodule.name] = ins

            if submodule.is_attn:
                head_acts = submodule.get_per_head_outputs()
                head_acts = submodule.handle_post_attn_mlp_norm(model, head_acts)
                for h in range(num_heads):
                    hidden_states_base[f"{submodule.name}_h{h}"] = head_acts[:,:,h,:].detach().save()
            else:
                act = submodule.get_out_activation()
                hidden_states_base[submodule.name] = act.detach().save()
        logits_base = model.lm_head.output.save()
    
    return hidden_states_base, logits_base, ln_ins_base

def forward_steer(model, submodules: list[Submodule], steer_submod: Submodule, inputs, steer_vec):
    num_heads = model.config.num_attention_heads
    hidden_states_steer = {}
    with t.no_grad(), model.trace(inputs):
        base_act = steer_submod.get_out_activation().clone()
        steered_act = base_act + steer_vec.to(steer_submod.device)
        steer_submod.set_out_activation(steered_act)
        for submodule in submodules:
            if submodule.is_attn:
                head_acts = submodule.get_per_head_outputs()
                head_acts = submodule.handle_post_attn_mlp_norm(model, head_acts)
                for h in range(num_heads):
                    hidden_states_steer[f"{submodule.name}_h{h}"] = head_acts[:,:,h,:].detach().save()
            else:
                act = submodule.get_out_activation()
                hidden_states_steer[submodule.name] = act.detach()
        logits_steer = model.lm_head.output.save()
    
    return hidden_states_steer, logits_steer

def evaluate_sample(model, submodules, resid_submod, sample, input_masks, node_names, steer_vec):

    inputs = sample['input_ids']
    prompt_len = sample['prompt_length']

    base_acts, logits_base, base_ln_ins = forward_base(model, submodules, inputs)
    steered_acts, logits_steer = forward_steer(model, submodules, resid_submod, inputs, steer_vec)

    logits_circuit = forward_with_circuit(
        model, submodules, resid_submod, steer_vec, inputs, input_masks, node_names, base_acts)

    base_matches, circuit_matches = metric_match_fn(logits_steer, logits_base, logits_circuit, prompt_len)
    faithfulness_kl = metric_kl1_fn(logits_steer, logits_base, logits_circuit, prompt_len)
    faithfulness_dir_kl = metric_kl2_fn(logits_steer, logits_base, logits_circuit, prompt_len)
    gen_prob_diff = metric_logprob_fn(logits_steer, logits_base, logits_circuit, prompt_len)
    return faithfulness_kl, faithfulness_dir_kl, base_matches, circuit_matches, gen_prob_diff


def run(examples, model, submodules, resid_submod, input_mask, node_names, steer_vec):
    batch_size = 1
    
    total_toks_generated = 0
    scores_kl_1 = []
    scores_kl_2 = []
    base_matches, circuit_matches = [], []
    gen_prob_diffs = []
    for sample in tqdm(examples):
        sample_f1, sample_f2, sample_base_matches, sample_circuit_matches, sample_gen_diff = evaluate_sample(model, submodules, resid_submod, sample, input_mask, node_names, steer_vec)
        bsz, num_generated = sample_f1.shape
        # print(sample_f1.shape, sample_f2.shape, sample_base_matches.shape, sample_circuit_matches.shape, sample_gen_diff.shape)

        scores_kl_1.append(sample_f1.flatten())
        scores_kl_2.append(sample_f2.flatten())
        base_matches.append(sample_base_matches.flatten().float())
        circuit_matches.append(sample_circuit_matches.flatten().float())
        gen_prob_diffs.append(sample_gen_diff.flatten())
        
        total_toks_generated += num_generated
    
    score_dict = {
        'kl_1': t.cat(scores_kl_1),
        'kl_2': t.cat(scores_kl_2),
        'base_matches': t.cat(base_matches),
        'circuit_matches': t.cat(circuit_matches),
        'gen_diff': t.cat(gen_prob_diffs),
    }

    return score_dict, total_toks_generated

def get_input_masks(edges, submodules: list, steer_submod: Submodule, num_attn_heads: int, num_kv_groups: int):
    """
    Returns input_mask shape num_downstream x num_upstream
    """
    steer_layer_idx = int(steer_submod.name.split('_')[1])+1
    # get upstream node names
    node_names = []
    for submodule in submodules:
        if submodule.is_attn:
            for h in range(num_attn_heads):
                node_names.append(f"{submodule.name}_h{h}")
        else:
            node_names.append(submodule.name)

    # set up mask
    input_masks = defaultdict(dict)
    for submodule in submodules:
        if submodule.is_attn:
            layer_idx = int(submodule.name.split('_')[1])
            # handle q
            for h in range(num_attn_heads):
                input_masks[f"{submodule.name}_h{h}_q"] = {
                    'mask': t.zeros(len(node_names)-1, dtype=t.bfloat16),
                    'slice': 1 + (layer_idx-steer_layer_idx)*(num_attn_heads+1)
                }

            # handle k
            for g in range(num_kv_groups):
                input_masks[f"{submodule.name}_h{g}_k"] = {
                    'mask': t.zeros(len(node_names)-1, dtype=t.bfloat16),
                    'slice': 1 + (layer_idx-steer_layer_idx)*(num_attn_heads+1)
                }

            # handle v
            for g in range(num_kv_groups):
                input_masks[f"{submodule.name}_h{g}_v"] = {
                    'mask': t.zeros(len(node_names)-1, dtype=t.bfloat16),
                    'slice': 1 + (layer_idx-steer_layer_idx)*(num_attn_heads+1)
                }
        elif submodule.is_mlp:
            input_masks[submodule.name] = {
                'mask': t.zeros(len(node_names)-1, dtype=t.bfloat16),
                'slice': 1 + num_attn_heads + (layer_idx-steer_layer_idx)*(num_attn_heads+1)
            }
        elif submodule.is_head:
            input_masks[submodule.name] = {
                'mask': t.zeros(len(node_names)-1, dtype=t.bfloat16),
                'slice': len(node_names)-1
            }
    
    def map_name_to_ind(name):
        for i, n in enumerate(node_names):
            if name == n:
                return i
        raise ValueError(f"missing name {name}")

    # populate mask
    for edge_d in edges:
        up, down, score = edge_d['up'], edge_d['down'], edge_d['score']
        input_masks[down]['mask'][map_name_to_ind(up)] = 1

    return input_masks, node_names


def invert_input_masks(input_masks):
    """
    Invert the circuit masks so that included edges become excluded and vice versa.
    This allows testing faithfulness on the complement of the circuit.
    """
    inverted = {}
    for key, val in input_masks.items():
        ref_mask = val['mask']
        ref_slice = val['slice']
        mask = t.zeros(len(ref_mask), dtype=t.bfloat16)
        mask[:ref_slice] = (1 - ref_mask[:ref_slice])
        inverted[key] = {
            'mask': mask,
            'slice': ref_slice,
        }
        # print(f"ref mask for {key}")
        # print(ref_mask)
        # print(f"invert mask for {key}")
        # print(mask)
    return inverted


if __name__ == '__main__':
    use_ln = False
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--learn_type', type=str, choices=['dim', 'ntp', 'reps', 'ortho'])
    parser.add_argument('--concept', type=str, default="refusal")
    parser.add_argument('--model_path', type=str, default="google/gemma-2-2b-it")
    parser.add_argument('--n', type=int, default=50)
    parser.add_argument('--layer', type=int, default=None)
    parser.add_argument('--method', type=str, default='simple', choices=['simple', 'backward', 'forward', 'random'])
    parser.add_argument('--max_examples', type=float, default=100)
    parser.add_argument('--harm_flag', action="store_true", default=False)
    parser.add_argument('--invert', action='store_true', default=False)
    parser.add_argument('--learn_path', type=str, default=None)
    parser.add_argument('--prefix_dir', type=str, default=None)
    args = parser.parse_args()
    print(f'learn type: {args.learn_type}, learn_path: {args.learn_path}')

    model_path = args.model_path
    exp_name = args.exp_name
    circuit_dir1 = f"circuits/{args.model_path.split('/')[-1]}/{exp_name}"
    circuit_dir2 = f"/fs/nexus-scratch/scheng03/steer-interp-results/{circuit_dir1}"
    if os.path.exists(f"{circuit_dir1}"):
        circuit_dir = circuit_dir1
    elif os.path.exists(f"{circuit_dir2}"):
        circuit_dir = circuit_dir2
    else:
        print(f"save dirs {circuit_dir1}, {circuit_dir2} do not exist")
    try:
        with open(os.path.join(circuit_dir, "config.json"), 'r') as configfile:
            config = Config.from_dict(json.load(configfile))
        print(f'loaded circuit with harm_flag {config.harm_flag}, steer_flag {config.steer_flag}, metric {config.metric}')
    except Exception as e:
        print('could not find config, assuming this is an aggregated exp')
    concept_params = {
        'refusal': {'harm_flag': args.harm_flag}
    }[args.concept]

    layer = dim_layer_map[model_path] if args.layer is None else args.layer
    layer, pos = dim_layer_pos_dict[model_path][layer]
    print(f'Layer {layer}, Position {pos}')
    

    print(f"Using steered generations on {'harmful' if args.harm_flag else 'harmless'} data")
    
    # load steer vector
    learned = True if args.learn_type != 'dim' else False
    learn_path = None if (args.learn_path is None or args.learn_path == 'None') else args.learn_path
    vector, coeff, layer_idx = load_steering_vector(
        args.concept, 
        model_path, 
        concept_params, 
        pos, 
        layer,
        learned=learned,
        vector_base_path=learn_path
    )
    steer_vec = coeff*vector.to(t.bfloat16)

    # load model and submodules
    model = LanguageModel(model_path, dtype=t.bfloat16, device_map="auto", dispatch=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    submodules = load_submodules(model, True)
    steer_submodules, resid_submod = get_steer_submodules(submodules, layer_idx)

    # load examples
    examples = prep_data(model_path, concept_params, tokenizer, pos, layer, learned, args.learn_path)
    max_examples = args.max_examples
    if len(examples) > max_examples:
        examples = examples[:max_examples]
        print(f'using the first {max_examples} examples')
    print(f'num examples: {len(examples)}')

    # load edges
    files = [f for f in os.listdir(circuit_dir) if f.startswith(f"edges_asked{args.n}_{args.method}") and f.endswith(".json")]
    assert len(files) == 1, f"{files}"
    edgefile = files[0]
    with open(os.path.join(circuit_dir, edgefile), 'r') as edgesfile:
        print(f'Opening: {edgefile}')
        edges = json.load(edgesfile)

    input_masks, node_names = get_input_masks(edges, steer_submodules, resid_submod, model.config.num_attention_heads, model.config.num_key_value_heads)
    if args.invert:
        input_masks = invert_input_masks(input_masks)

    scores, total_toks_generated = run(examples, model, steer_submodules, resid_submod, input_masks, node_names, steer_vec)
    base_steer_different = ~scores['base_matches'].bool()

    score_stats = {
        'num_toks': total_toks_generated,
        'num_examples': len(examples),
        'harm_flag': args.harm_flag,
        'base_matches': {
            'mean': scores['base_matches'].mean().item(),
            'std': scores['base_matches'].std().item(),
        }
    }

    # first handle metrics that are per token per sample
    for metric in ['kl_1', 'kl_2', 'circuit_matches']:
        score_stats[metric] = {
            'overall': {
                'mean': scores[metric].mean().item(), 
                'std': scores[metric].std().item()
            },
            'base_steer_differ': {
                'mean': scores[metric][base_steer_different].mean().item(), 
                'std': scores[metric][base_steer_different].std().item()
            },
            'base_steer_same': {
                'mean': scores[metric][~base_steer_different].mean().item(), 
                'std': scores[metric][~base_steer_different].std().item()
            }
        }

    # handle metric that is per token per sample
    score_stats['gen_diff'] = {
        'overall': {
            'mean': scores['gen_diff'].mean().item(),
            'std': scores['gen_diff'].std().item(),
        }
    }

    save_prefix_dir = args.prefix_dir
    if save_prefix_dir is None or save_prefix_dir == 'None':
        save_dir = circuit_dir
    else:
        save_dir = os.path.join(circuit_dir, save_prefix_dir)
        os.makedirs(save_dir, exist_ok=True)
    stats_f_name = f"faithfulness_{args.n}_{args.method}_{'harmful' if args.harm_flag else 'harmless'}{'_invert' if args.invert else ''}.json"

    with open(os.path.join(save_dir, stats_f_name), 'w') as f:
        json.dump(score_stats, f, indent=4)

    print(f'Saved results to {os.path.join(save_dir, stats_f_name)}')
    