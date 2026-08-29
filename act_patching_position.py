"""
Identify important components using activation patching method
"""
import torch as t
import torch.nn.functional as F
import os
from nnsight import LanguageModel
from tqdm import tqdm
import pandas as pd
import math
import gc
import numpy as np
import json
from collections import defaultdict
from functools import partial
from config_position import Config, parse_args

from transformers import GPTNeoXForCausalLM, AutoModelForCausalLM, AutoTokenizer, GemmaForCausalLM, LlamaForCausalLM

from patching.loading_utils import load_submodules
from patching.attribution import patching_effect, compute_edge_simple, compute_edge_with_ln_slow, obtain_ln_grads
from utils.data_utils import load_steering_vector
from utils.gen_utils import load_model_and_tokenizer, set_seeds
from data_preprocess import get_steer_hooks, fwd_with_hooks

set_seeds()

def get_circuit(
    inputs, model, steer_vec, layer_idx, submodule_stash, metric_fn,
    config, src_intervention_fn, dest_intervention_fn
):

    # first get the patching effect of everything on y
    effect_out, steer_submodules = patching_effect(
        inputs,
        model,
        steer_vec,
        layer_idx,
        submodule_stash,
        metric_fn,
        config,
        src_intervention_fn,
        dest_intervention_fn,
    )
    node_effects = effect_out.effects
    deltas = effect_out.deltas
    grads = effect_out.grads
    out_grads = effect_out.out_grads
    total_effect = effect_out.total_effect
    head_effects = effect_out.head_effects
    head_deltas = effect_out.head_deltas
    head_out_grads = effect_out.head_out_grads
    qkv_in_grads = effect_out.qkv_in_grads
    edge_effects = effect_out.edge_effects

    if config.method == 'exact':
        node_effects['y'] = total_effect
        assert edge_effects is not None
        return node_effects, edge_effects, None, None
    
    nodes = {"y": total_effect} # should be the kl divergence of important tokens averaged over the batch
    for submod in steer_submodules:
        if submod.is_attn:
            b, s, n_heads, d = head_effects[submod].shape
            for h in range(n_heads):
                nodes[f"{submod.name}_h{h}"] = head_effects[submod][:,:,h,:]
        else:
            nodes[submod.name] = node_effects[submod]
    
    if config.nodes_only:
        if config.method == "exact":
            pass
        if config.aggregation == "sum":
            for k in nodes:
                if k != "y":
                    nodes[k] = nodes[k].sum(dim=1)
        nodes = {k: v.mean(dim=0) for k, v in nodes.items()}
        return nodes, None
    
    edges = defaultdict(lambda: {})

    if config.use_ln_grad:
        assert config.save_grads # needs concatenated grads
        grads, qkv_in_grads = obtain_ln_grads(inputs, model, steer_submodules, grads, qkv_in_grads, steer_vec, layer_idx, config.trapezoidal)

    def handle_downstream():
        for j, downstream in enumerate(steer_submodules[i+1:], start=i+1):
            midstream = [steer_submodules[k] for k in range(i+1, j)]
            down_name = downstream.name
            if downstream.is_attn:
                for l in 'qkv':
                    down_grads = qkv_in_grads[downstream][l]
                    (edge_effect, grad) = compute_edge_simple(up_deltas, down_grads, config.save_grads)
                    edges[up_name][f"{down_name}_{l}"] = edge_effect
            else:
                down_grads = grads[downstream]
                (edge_effect, grad) = compute_edge_simple(up_deltas, down_grads, config.save_grads)
                edges[up_name][down_name] = edge_effect
    
    # now we work backward through the model to get the edges
    for i, upstream in enumerate(steer_submodules):
        # accumulated_grad = t.zeros_like(out_grads[upstream])
        if upstream.is_attn:
            num_heads = head_deltas[upstream].shape[2] # b s h d
            for h in range(num_heads):
                up_deltas = head_deltas[upstream][:,:,h,:]
                up_name = f"{upstream.name}_h{h}"
                handle_downstream()
        else:
            up_deltas = deltas[upstream]
            up_name = upstream.name
            handle_downstream()

    if config.aggregation == "none":
        # aggregate across sequence position by summing
        for up in edges:
            for down in edges[up]:
                weight_matrix = edges[up][down] # b s d
                edges[up][down] = weight_matrix.mean(dim=0)
        for node in nodes:
            if node != "y":
                val = nodes[node] # b s d
                nodes[node] = val.mean(dim=0)
    else:
        raise ValueError(f"Unknown aggregation: {config.aggregation}")
    # if refactoring, could do mean across batch separately outside of aggregation above

    if config.save_grads:
        returned_out_grads = {}
        returned_in_grads = {}
        for submodule in steer_submodules:
            if submodule.is_attn:
                n_steps, b, s, num_heads, d = head_out_grads[submodule].shape
                for h in range(num_heads):
                    returned_out_grads[f"{submodule.name}_h{h}"] = head_out_grads[submodule][:,:,:,h,:].sum(dim=2).mean(dim=1)
                for l in "qkv":
                    n_steps, b, s, d = qkv_in_grads[submodule][l].shape
                    returned_in_grads[f"{submodule.name}_{l}"] = qkv_in_grads[submodule][l].sum(dim=2).mean(dim=1)
            else:
                n_steps, b, s, d = out_grads[submodule].shape
                returned_out_grads[submodule.name] = out_grads[submodule].sum(dim=2).mean(dim=1)
                returned_in_grads[submodule.name] = grads[submodule].sum(dim=2).mean(dim=1)
    else:
        returned_out_grads = None
        returned_in_grads = None
    
    del effect_out
    return nodes, edges, returned_out_grads, returned_in_grads

if __name__ == "__main__":
    # set params
    config = parse_args()

    print(f"Using model {config.model_path}")

    # obtain steering vector
    vector, coeff, layer_idx = load_steering_vector(config.concept, config.model_path, config.concept_params)
    vector = vector.to(t.bfloat16)
    print(f'steering vector applied at layer {layer_idx}, with dtype {vector.dtype}, coeff {coeff}')

    # assemble instructions
    kl_data_path = config.kl_data_path
    print(kl_data_path)
    df = pd.read_parquet(kl_data_path)
    examples = []

    # obtain positions for the kl thresholds
    sanity_check, atol_check = False, 0.1
    model, tokenizer = load_model_and_tokenizer(config.model_path)
    for row in df.itertuples(index=False):
        tokens = t.tensor(row.tokens)
        prompt_len = row.prompt_length
        kl_divs = row.kl_divergence
        mismatch = row.mismatch

        prompt_inputs = {
            'input_ids': tokens[:prompt_len].unsqueeze(0),
            'attention_mask': t.ones((1, len(tokens[:prompt_len])), dtype=t.long)
        }
        prompt_outputs_base = fwd_with_hooks(model, prompt_inputs, fwd_hooks=[], fwd_pre_hooks=[])
        prompt_logits_base = prompt_outputs_base.logits.detach().float()
        fwd_hooks, fwd_pre_hooks = get_steer_hooks(model, layer_idx, vector, coeff)
        prompt_outputs_steer = fwd_with_hooks(model, prompt_inputs, fwd_hooks=fwd_hooks, fwd_pre_hooks=fwd_pre_hooks)
        prompt_logits_steer = prompt_outputs_steer.logits.detach().float()
        base_y = t.argmax(prompt_logits_base[0,-1], dim=-1)
        steer_y = t.argmax(prompt_logits_steer[0,-1], dim=-1)
        mismatch = base_y != steer_y

        if sanity_check: # check if the kl divs line up correctly
            whole_convo_inputs = {
                'input_ids': tokens.unsqueeze(0),
                'attention_mask': t.ones((1, len(tokens)), dtype=t.long)
            }
            whole_outputs_base = fwd_with_hooks(model, whole_convo_inputs, fwd_hooks=[], fwd_pre_hooks=[])
            whole_logits_base = whole_outputs_base.logits.detach().float()
            fwd_hooks, fwd_pre_hooks = get_steer_hooks(model, layer_idx, vector, coeff)
            whole_outputs_steer = fwd_with_hooks(model, whole_convo_inputs, fwd_hooks=fwd_hooks, fwd_pre_hooks=fwd_pre_hooks)
            whole_logits_steer = whole_outputs_steer.logits.detach().float()

            convo_inputs = {
                'input_ids': tokens.unsqueeze(0),
                'attention_mask': t.ones((1, len(tokens)), dtype=t.long)
            }
            if config.metric == 'logit':
                base_ys = t.argmax(whole_logits_base, dim=-1)
                steer_ys = t.argmax(whole_logits_steer, dim=-1)
                mismatch_sanity = (base_ys != steer_ys).cpu().numpy()
                assert (mismatch == mismatch_sanity).all()
            elif config.metric in ['dirKL', 'nodirKL']:
                if config.steer_flag:
                    # using steered sample: checking divergence of no steer probability against steer probability
                    log_probs_input = F.log_softmax(whole_logits_base, dim=-1)
                    probs_target = F.softmax(whole_logits_steer, dim=-1)
                else:
                    log_probs_input = F.log_softmax(whole_logits_steer, dim=-1)
                    probs_target = F.softmax(whole_logits_base, dim=-1)
                kl_divs_sanity = F.kl_div(log_probs_input, probs_target, reduction='none').sum(dim=-1)
                kl_divs_sanity = t.clamp_min(kl_divs_sanity, 0).cpu().numpy()
                assert np.allclose(kl_divs, kl_divs_sanity, atol=atol_check), print(kl_divs, kl_divs_sanity, tokenizer.decode(tokens))

        if mismatch:
            examples.append({
                'query': tokens[:prompt_len],
                'tok_steer': steer_y,
                'tok_base': base_y,
                'strings': tokenizer.decode(tokens[:prompt_len]),
                'answer_steer': tokenizer.decode(steer_y),
                'answer_base': tokenizer.decode(base_y),
                'pois': [-1]
            })
    if sanity_check:
        del model, tokenizer


    num_examples = min([config.num_examples, len(examples)])

    if num_examples < num_examples:  # warn the user
        print(
            f"Total number of examples is less than {num_examples}. Using {num_examples} examples instead."
        )
    print('done loading data')
    batch_size = 1
    n_batches = math.ceil(num_examples / batch_size)
    batches = [
        examples[batch * batch_size : (batch + 1) * batch_size]
        for batch in range(n_batches)
    ]

    print("computing circuit")

    model = LanguageModel(config.model_path, device_map="auto", dispatch=True, dtype=t.bfloat16)
    if not model.model.is_gradient_checkpointing:
        print('setting gradient checkpointing')
        model.model.gradient_checkpointing_enable()
        model.model.config.use_cache=False
    print(f'loading model {config.model_path} with {len(model.model.layers)} layers and device map \n{model.hf_device_map}')

    submodules = load_submodules(
        model,
        separate_by_type=True,
    )
    if config.check_negs:
        assert config.method == 'exact'
        assert os.path.exists(f"{config.save_path}")
        assert os.path.exists(f"{os.path.splitext(config.save_path)[0]}_neg_circuit_components.json"), "need to run evaluate_circuit.py"
        with open(f"{os.path.splitext(config.save_path)[0]}_neg_circuit_components.json") as f:
            neg_circuit_components = json.load(f)

    running_nodes = []
    running_edges = []
    running_out_grads = []
    running_in_grads = []
    
    def get_metric_fn(metric_type):
        def metric_logit_fn(logits_src, logits_dest, logits_patch):

            src_ys = t.argmax(logits_src[:,-1], dim=-1)
            dest_ys = t.argmax(logits_dest[:,-1], dim=-1)

            bsz = len(logits_src)
            loi_approx_src_toks = logits_patch[t.arange(bsz), -1, src_ys]
            loi_approx_dest_toks = logits_patch[t.arange(bsz), -1, dest_ys]

            m = loi_approx_src_toks - loi_approx_dest_toks
            return m.mean(dim=0)
        
        def metric_kl_fn(logits_src, logits_dest, logits_patch):
            def prepare_loi(logits):
                ret = t.concat([logits[i, -1] for i in range(len(logits))]).float()
                return ret
            orig_dtype = logits_src.dtype
            loi_approx = prepare_loi(logits_patch)
            loi_src = prepare_loi(logits_src)
            loi_dest = prepare_loi(logits_dest)

            log_prob_approx = F.log_softmax(loi_approx, dim=-1)
            prob_src = F.softmax(loi_src, dim=-1)
            prob_dest = F.softmax(loi_dest, dim=-1)

            if metric_type == 'dirKL':
                kl_divs = F.kl_div(log_prob_approx, prob_dest, reduction='batchmean') - F.kl_div(log_prob_approx, prob_src, reduction='batchmean')
            elif metric_type == 'nodirKL':
                kl_divs = F.kl_div(log_prob_approx, prob_dest, reduction='batchmean')
            return kl_divs.to(orig_dtype)
        if metric_type == 'logit':
            print('using logit metric function')
            return metric_logit_fn
        elif metric_type in ['dirKL', 'nodirKL']:
            print('using KL metric function')
            return metric_kl_fn
        else:
            raise ValueError(f'missing metric type {metric_type}')
    
    def get_src_and_dest_fns(steer_flag):
        def add_steering(submodule, steer_vec):
            resid_base_act = submodule.get_out_activation().clone()
            resid_steer_act = resid_base_act + steer_vec.to(submodule.device)
            submodule.set_out_activation(resid_steer_act)
        
        def no_steering(submodule, steer_vec):
            pass
        
        if steer_flag: # the completion uses the steered model, so set the steered model as the destination
            dest_intervention_fn = add_steering
            src_intervention_fn = no_steering
        else:
            dest_intervention_fn = no_steering
            src_intervention_fn = add_steering
        return src_intervention_fn, dest_intervention_fn
    
    total_num_pos = 0
    test_examples = []
    test_decoded = []
    steer_ys = []
    base_ys = []
    for batch in tqdm(batches, desc="Batches"):
        assert len(batch) == 1, "only support single example for accuracy"

        inputs = t.cat([e["query"].unsqueeze(0) for e in batch])
        total_num_pos += 1
        batch_decoded = []
        for ex in batch:
            decoded = []
            for tok in ex['query']:
                decoded.append(tokenizer.decode(tok))
            batch_decoded.append(decoded)
        if len(batch_decoded) == 1: batch_decoded = batch_decoded[0]
        test_decoded.append(batch_decoded)
        test_examples.append(t.cat([e['query'].unsqueeze(0) for e in batch]))
        steer_ys.append(t.cat([e['tok_steer'].unsqueeze(0) for e in batch]))
        base_ys.append(t.cat([e['tok_base'].unsqueeze(0) for e in batch]))

        src_intervention_fn, dest_intervention_fn = get_src_and_dest_fns(config.steer_flag)

        nodes, edges, out_grads, in_grads = get_circuit(
            inputs,
            model,
            vector*coeff,
            layer_idx,
            submodules,
            get_metric_fn(config.metric),
            # partial(metric_fn, kl_poi=poi),
            config,
            src_intervention_fn,
            dest_intervention_fn,
        )
        
        running_nodes.append({
            k: nodes[k].detach().cpu() for k in nodes.keys()
        })
        if config.save_grads:
            running_out_grads.append({
                k: out_grads[k].detach().cpu() for k in out_grads.keys()
            })
            running_in_grads.append({
                k: in_grads[k].detach().cpu() for k in in_grads.keys()
            })
        if not config.nodes_only:
            running_edges.append({
                k: {
                    kk: edges[k][kk].detach().cpu()
                    for kk in edges[k].keys()
                }
                for k in edges.keys()
            })
            
        for key in nodes:
            assert t.isnan(nodes[key]).sum() == 0, "smth fucked up happened"
        
        model.zero_grad()
        del nodes, edges, out_grads, in_grads
        gc.collect()
        t.cuda.empty_cache()
        break

    save_dict = {
        "examples": test_examples, 
        "examples_decoded": test_decoded,
        "base_ys": base_ys,
        "steer_ys": steer_ys,
        "num_positions": total_num_pos, 
        "nodes": running_nodes, 
        "edges": running_edges,
        "out_grads": running_out_grads,
        "in_grads": running_in_grads}
    
    config.update_save_dir()
    os.makedirs(config.save_dir, exist_ok=True)
    with open(os.path.join(config.save_dir, "patching_results.pt"), "wb") as outfile:
        t.save(save_dict, outfile)
    with open(os.path.join(config.save_dir, "config.json"), "w") as configfile:
        json.dump(config.to_dict(), configfile, indent=4)
    print(f'saved patching results to {config.save_dir}')
