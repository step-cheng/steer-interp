from collections import namedtuple
import torch as t
from tqdm import tqdm
from collections import defaultdict
from numpy import ndindex
from .loading_utils import Submodule, SubmoduleStash
from nnsight.intervention.envoy import Envoy
from dictionary_learning.dictionary import Dictionary, JumpReluAutoEncoder
from typing import Callable, Union, Optional
import types
from functools import partial
import gc

EffectOut = namedtuple('EffectOut', ['effects', 'deltas', 'grads', 'out_grads', 'total_effect', 'head_effects', 'head_deltas', 'head_out_grads', 'qkv_in_grads', 'edge_effects'])

def get_steer_submodules(submodules: SubmoduleStash, steer_layer_idx):
    resid_submod = submodules.resids[steer_layer_idx-1]
    output_submodules: list[Submodule] = [resid_submod] # because steering is applied as a posthook...
    for j in range(steer_layer_idx, len(submodules.attns)):
        output_submodules.append(submodules.attns[j])
        output_submodules.append(submodules.mlps[j])
    output_submodules.append(submodules.lm_head)
    return output_submodules, resid_submod

def get_unused_submodules(submodules: SubmoduleStash, steer_layer_idx):
    unused_submodules: list[Submodule] = [submodules.embed]
    for i in range(steer_layer_idx-1):
        unused_submodules.append(submodules.attns[i])
        unused_submodules.append(submodules.mlps[i])
        unused_submodules.append(submodules.resids[i])
    return unused_submodules

def _pe_attrib(
        clean,
        patch,
        model,
        submodules: list[Submodule],
        dictionaries: dict[Submodule, Dictionary],
        metric_fn,
        metric_kwargs=dict(),
):
    
    hidden_states_clean = {}
    grads = {}
    with model.trace(clean):
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            x = submodule.get_activation()
            x_hat, f = dictionary(x, output_features=True) # x_hat implicitly depends on f
            residual = x - x_hat
            hidden_states_clean[submodule] = SparseAct(act=f, res=residual).save()
            grads[submodule] = hidden_states_clean[submodule].grad.save()
            residual.grad = t.zeros_like(residual)
            x_recon = x_hat + residual
            submodule.set_activation(x_recon)
            x.grad = x_recon.grad
        metric_clean = metric_fn(model, **metric_kwargs).save()
        metric_clean.sum().backward()
    hidden_states_clean = {k : v.value for k, v in hidden_states_clean.items()}
    grads = {k : v.value for k, v in grads.items()}

    if patch is None:
        hidden_states_patch = {
            k : SparseAct(act=t.zeros_like(v.act), res=t.zeros_like(v.res)) for k, v in hidden_states_clean.items()
        }
        total_effect = None
    else:
        hidden_states_patch = {}
        with t.no_grad(), model.trace(patch):
            for submodule in submodules:
                dictionary = dictionaries[submodule]
                x = submodule.get_activation()
                x_hat, f = dictionary(x, output_features=True)
                residual = x - x_hat
                hidden_states_patch[submodule] = SparseAct(act=f, res=residual).save()
            metric_patch = metric_fn(model, **metric_kwargs).save()
        total_effect = (metric_patch.value - metric_clean.value).detach()
        hidden_states_patch = {k : v.value for k, v in hidden_states_patch.items()}

    effects = {}
    deltas = {}
    for submodule in submodules:
        patch_state, clean_state, grad = hidden_states_patch[submodule], hidden_states_clean[submodule], grads[submodule]
        delta = patch_state - clean_state.detach() if patch_state is not None else -clean_state.detach()
        effect = delta @ grad
        effects[submodule] = effect
        deltas[submodule] = delta
        grads[submodule] = grad
    total_effect = total_effect if total_effect is not None else None
    
    return EffectOut(effects, deltas, grads, total_effect)

def _pe_exact(
    inputs,
    model,
    steer_vec,
    layer_idx,
    submodules: list[tuple[Submodule, int]],
    metric_fn,
    src_intervention_fn,
    dst_intervention_fn,
):
    """
    submodules: should contain list of specific submodules for patching. 
    """
    steer_submodules, resid_submod = get_steer_submodules(submodules, layer_idx)
    num_heads = model.config.num_attention_heads
    bsz, sequence_len = inputs.shape

    num_heads = model.config.num_attention_heads
    hidden_states_src = {}
    head_outputs_src = {}
    with t.no_grad(), model.trace(inputs):
        src_intervention_fn(resid_submod, steer_vec)
        for submodule in steer_submodules:
            if submodule.is_attn:
                head_out = submodule.get_per_head_outputs()
                head_outputs_src[submodule] = head_out.save()

            x = submodule.get_out_activation()
            hidden_states_src[submodule] = x.save()
        logits_src = model.lm_head.output.save()

    hidden_states_dst = {}
    head_outputs_dst = {}
    with t.no_grad(), model.trace(inputs):
        dst_intervention_fn(resid_submod, steer_vec)
        for submodule in steer_submodules:
            if submodule.is_attn:
                head_out = submodule.get_per_head_outputs()
                head_outputs_dst[submodule] = head_out.save()

            x_out = submodule.get_out_activation()
            hidden_states_dst[submodule] = x_out.save()
        logits_dst = model.lm_head.output.save()

    metric_src = metric_fn(logits_src, logits_dst, logits_src)
    metric_dst = metric_fn(logits_src, logits_dst, logits_dst)
    total_effect = metric_src - metric_dst

    deltas = {}
    head_deltas = {}
    for submodule in steer_submodules:
        delta = hidden_states_src[submodule] - hidden_states_dst[submodule]
        deltas[submodule] = delta.detach()
        
        if submodule.is_attn:
            head_delta = head_outputs_src[submodule] - head_outputs_dst[submodule]
            head_deltas[submodule] = head_delta.detach()
    
    # node patching, per position
    print(f'starting node patching')
    node_effects = {}
    for submodule in steer_submodules:
        print(submodule.name)
        if submodule.is_attn:
            src_head_state = head_outputs_src[submodule]
            for h in range(num_heads):
                node_effects[f"{submodule.name}_h"] = t.zeros(sequence_len, device=model.device)
                for pos in range(sequence_len):
                    with t.no_grad(), model.trace(inputs):
                        dst_intervention_fn(resid_submod, steer_vec)
                        new_act = src_head_state[:,pos,h,:]
                        head_acts = submodule.get_per_head_outputs().detach()
                        head_acts[:,pos,h,:] = new_act
                        submodule.replace_attn_output_with_head_sum(head_acts)
                        logits_patch = model.lm_head.output
                    
                        metric_patch = metric_fn(logits_src, logits_dst, logits_patch).save()
                    node_effects[f"{submodule.name}_h"][pos] = metric_patch - metric_dst
        else:
            patch_state = hidden_states_src[submodule]
            node_effects[f"{submodule.name}"] = t.zeros(sequence_len, device=model.device)
            for pos in range(sequence_len):
                with t.no_grad(), model.trace(inputs):
                    dst_intervention_fn(resid_submod, steer_vec)
                    new_act = patch_state[:,pos,:]
                    submodule.set_out_activation(new_act)

                    logits_patch = model.lm_head.output
                
                    metric_patch = metric_fn(logits_src, logits_dst, logits_patch).save()
                node_effects[f"{submodule.name}"][pos] = metric_patch - metric_dst
    
    print('finished node patching')
    # edge effects, per position

    edge_effects = defaultdict(dict)
    for up_idx, up_submod in enumerate(steer_submodules[:-1]):
        print(up_submod.name)
        if up_submod.is_attn:
            for h in range(num_heads):
                for down_submod in steer_submodules[up_idx+1:]:
                    ln_mod = down_submod.obtain_pre_ln_module()
                    if down_submod.is_attn:
                        patch_delta = head_deltas[up_submod][:,:,h,:]
                        with t.no_grad(), model.trace(inputs) as tracer:
                            dst_intervention_fn(resid_submod, steer_vec)
                            ln_in_act = ln_mod.get_in_activation().clone()
                            ln_in_act += patch_delta
                            ln_mod.set_in_activation(ln_in_act)
                            new_acts = ln_mod.get_out_activation().save()
                            tracer.stop()
                        for i, l in enumerate("qkv"):
                            ee = t.zeros(sequence_len)
                            for pos in range(sequence_len):
                                with t.no_grad(), model.trace(inputs):
                                    dst_intervention_fn(resid_submod, steer_vec)
                                    in_acts = down_submod.get_qkv_inputs_separate() # q, k, v
                                    in_acts[i][:,pos,:] = new_acts[:, pos,:]
                                    down_submod.set_qkv_inputs_separate(*in_acts)
                                    logits_patch = model.lm_head.output
                        
                                    metric = metric_fn(logits_src, logits_dst, logits_patch).save()
                                ee[pos] = metric - metric_dst
                            edge_effects[f"{up_submod.name}_h{h}"][f"{down_submod.name}_{l}"] = ee
                    else:
                        ln_mod = down_submod.obtain_pre_ln_from_module(model)
                        ee = t.zeros(sequence_len)
                        for pos in range(sequence_len):
                            patch_delta = head_deltas[up_submod][:,pos,h,:]
                            with t.no_grad(), model.trace(inputs):
                                dst_intervention_fn(resid_submod, steer_vec)
                                ln_in_act = ln_mod.get_in_activation().clone()
                                ln_in_act[:, pos,:] += patch_delta
                                ln_mod.set_in_activation(ln_in_act)

                                logits_patch = model.lm_head.output
                    
                                metric = metric_fn(logits_src, logits_dst, logits_patch).save()
                            ee[pos] = metric - metric_dst
                        edge_effects[f"{up_submod.name}_h{h}"][f"{down_submod.name}"] = ee
        else:
            for down_submod in steer_submodules[up_idx+1:]:
                ln_mod = down_submod.obtain_pre_ln_from_module(model)
                if down_submod.is_attn:
                    patch_delta = deltas[up_submod][:,:,:]
                    with t.no_grad(), model.trace(inputs) as tracer:
                        dst_intervention_fn(resid_submod, steer_vec)
                        ln_in_act = ln_mod.get_in_activation().clone()
                        ln_in_act += patch_delta
                        ln_mod.set_in_activation(ln_in_act)
                        new_acts = ln_mod.get_out_activation().save()
                        tracer.stop()
                    for i, l in enumerate("qkv"):
                        ee = t.zeros(sequence_len)
                        for pos in range(sequence_len):
                            with t.no_grad(), model.trace(inputs):
                                dst_intervention_fn(resid_submod, steer_vec)
                                in_acts = down_submod.get_qkv_inputs_separate() # q, k, v
                                in_acts[i][:,pos,:] = new_acts[:, pos,:]
                                down_submod.set_qkv_inputs_separate(*in_acts)
                                logits_patch = model.lm_head.output
                    
                                metric = metric_fn(logits_src, logits_dst, logits_patch).save()
                            ee[pos] = metric - metric_dst
                        edge_effects[f"{up_submod.name}"][f"{down_submod.name}_{l}"] = ee
                else:
                    ln_mod = down_submod.obtain_pre_ln_from_module(model)
                    ee = t.zeros(sequence_len)
                    for pos in range(sequence_len):
                        patch_delta = deltas[up_submod][:,pos,:]
                        with t.no_grad(), model.trace(inputs):
                            dst_intervention_fn(resid_submod, steer_vec)
                            ln_in_act = ln_mod.get_in_activation().clone()
                            ln_in_act[:, pos,:] += patch_delta
                            ln_mod.set_in_activation(ln_in_act)

                            logits_patch = model.lm_head.output
                
                            metric = metric_fn(logits_src, logits_dst, logits_patch).save()
                        ee[pos] = metric - metric_dst
                    edge_effects[f"{up_submod.name}"][f"{down_submod.name}"] = ee    

    return EffectOut(node_effects, deltas, None, None, total_effect, None, head_deltas, None, None, edge_effects), steer_submodules

def _pe_ig(
    inputs,
    model,
    steer_vec,
    layer_idx,
    all_submodules: SubmoduleStash,
    metric_fn,
    src_intervention_fn,
    dest_intervention_fn,
    steps=10,
    trapezoidal=True,
    save_grads=False,
):
    """
    Obtains gradients wrt inputs, whereas _pe_ig obtains gradients wrt outputs. Thus, no need for SparseAct
    """
    steer_submodules, resid_submod = get_steer_submodules(all_submodules, layer_idx)
    num_heads = model.config.num_attention_heads
    unused_submodules = get_unused_submodules(all_submodules, layer_idx)

    # cache src activations
    hidden_states_src = {}
    head_outputs_src = {}
    with t.no_grad(), model.trace(inputs):
        src_intervention_fn(resid_submod, steer_vec)
        for submodule in steer_submodules:
            if submodule.is_attn:
                head_out = submodule.get_per_head_outputs()
                head_out = submodule.handle_post_attn_mlp_norm(model, head_out)
                head_outputs_src[submodule] = head_out.save()

            x_out = submodule.get_out_activation()
            hidden_states_src[submodule] = x_out.save()
        logits_src = model.lm_head.output.save()

    # cache dest activation
    hidden_states_dest = {}
    head_outputs_dest = {}
    with t.no_grad(), model.trace(inputs):
        dest_intervention_fn(resid_submod, steer_vec)
        for submodule in steer_submodules:
            if submodule.is_attn:
                head_out = submodule.get_per_head_outputs()
                head_out = submodule.handle_post_attn_mlp_norm(model, head_out)
                head_outputs_dest[submodule] = head_out.save()

            x_out = submodule.get_out_activation()
            hidden_states_dest[submodule] = x_out.save()
        logits_dest = model.lm_head.output.save()
    
    metric_src = metric_fn(logits_src, logits_dest, logits_src)
    metric_dest = metric_fn(logits_src, logits_dest, logits_dest)
    total_effect = metric_src - metric_dest

    # obtain activation differences at the output of each submodule
    deltas = {}
    head_deltas = {}
    for submodule in steer_submodules:
        delta = hidden_states_src[submodule] - hidden_states_dest[submodule]
        deltas[submodule] = delta.detach().cpu()

        if submodule.is_attn:
            head_delta = head_outputs_src[submodule] - head_outputs_dest[submodule]
            head_deltas[submodule] = head_delta.detach().cpu()
    
    del hidden_states_dest, hidden_states_src
    
    in_grads = {}
    out_grads = {}
    head_out_grads = {}
    qkv_in_grads = {}

    if save_grads:
        accumulated_in_grads = {submod: [] for submod in steer_submodules}
        accumulated_out_grads = {submod: [] for submod in steer_submodules}
        accumulated_head_out_grads = {submod: [] for submod in steer_submodules}
        accumulated_qkv_in_grads = {submod: {'q': [], 'k': [], 'v': []} for submod in steer_submodules}
        def add_to_dict(data_dict, submod, data, key=None):
            if key: data_dict[submod][key].append(data.unsqueeze(0))
            else: data_dict[submod].append(data.unsqueeze(0))
    else:
        accumulated_in_grads = {submod: None 
                                for submod in steer_submodules}
        accumulated_out_grads = {submod: None
                                for submod in steer_submodules}
        accumulated_head_out_grads = {submod: None
                                    for submod in steer_submodules if submod.is_attn}
        accumulated_qkv_in_grads = {s: {'q': None, 'k': None, 'v': None} 
                                    for s in steer_submodules if s.is_attn}
        def add_to_dict(data_dict, submod, data, key=None):
            if key: 
                if data_dict[submod][key] is None: data_dict[submod][key] = data / steps
                else: data_dict[submod][key] += data / steps
            else: 
                if data_dict[submod] is None: data_dict[submod] = data / steps
                else: data_dict[submod] += data / steps

    for step in range(0,steps):
        alpha = (step+0.5) / steps if trapezoidal else step / steps
        input_acts = {}
        output_acts = {}
        head_output_acts = {}
        qkv_acts = {}

        with model.trace(inputs):
            # do not track gradients from submodules before steering.
            for submodule in unused_submodules:
                x_out = submodule.get_out_activation()
                submodule.set_out_activation(x_out.detach())
            for submodule in steer_submodules:
                x_in = submodule.get_in_activation()
                x_in.requires_grad_().retain_grad()
                input_acts[submodule] = x_in.save()

                if submodule == resid_submod:
                    # apply steering
                    x_dummy = submodule.get_out_activation().clone()
                    new_activation = x_dummy + alpha*steer_vec.to(submodule.device)
                    new_activation.requires_grad_().retain_grad()
                    submodule.set_out_activation(new_activation)
                    output_acts[submodule] = new_activation.save()
                elif submodule.is_attn:
                    # Separate Q/K/V inputs for gradient tracking
                    q_in, k_in, v_in = submodule.get_qkv_inputs_separate()
                    q_in.requires_grad_().retain_grad()
                    k_in.requires_grad_().retain_grad()
                    v_in.requires_grad_().retain_grad()
                    submodule.set_qkv_inputs_separate(q_in, k_in, v_in)
                    qkv_acts[submodule] = {
                        'q': q_in.save(),
                        'k': k_in.save(),
                        'v': v_in.save()
                    }
                    
                    # Per-head outputs with gradient routing
                    head_out = submodule.get_per_head_outputs()
                    head_out = submodule.handle_post_attn_mlp_norm(model, head_out)
                    head_output_acts[submodule] = head_out.save()
                    
                    # Also track full attention output to get out grads
                    x_out = submodule.get_out_activation()
                    x_out.requires_grad_().retain_grad()
                    output_acts[submodule] = x_out.save()
                else:
                    x_out = submodule.get_out_activation()
                    x_out.requires_grad_().retain_grad()
                    output_acts[submodule] = x_out.save()

            logits_mid = model.lm_head.output.save()
        metric = metric_fn(logits_src, logits_dest, logits_mid)
        metric.backward()

        for submodule in steer_submodules:
            grad_in = input_acts[submodule].grad.detach().cpu()
            grad_out = output_acts[submodule].grad.detach().cpu()
            add_to_dict(accumulated_in_grads, submodule, grad_in)
            add_to_dict(accumulated_out_grads, submodule, grad_out)
            
            if submodule.is_attn:
                for l in 'qkv':
                    l_grad = qkv_acts[submodule][l].grad.detach().cpu()
                    add_to_dict(accumulated_qkv_in_grads, submodule, l_grad, l)

                head_grad = submodule.get_head_grads(
                    output_acts[submodule].grad
                ).detach().cpu()
                add_to_dict(accumulated_head_out_grads, submodule, head_grad)

        model.zero_grad()
        del metric, logits_mid, input_acts, output_acts, head_output_acts, qkv_acts
        # t.cuda.empty_cache()
        
    effects = {}
    head_effects = {}
    # populate effects, delta, and grad
    for submodule in steer_submodules:
        if not submodule.is_attn:
            all_grads_in = t.cat(accumulated_in_grads[submodule], dim=0) if save_grads else accumulated_in_grads[submodule]
            all_grads_out = t.cat(accumulated_out_grads[submodule], dim=0) if save_grads else accumulated_out_grads[submodule]
            in_grads[submodule] = all_grads_in
            out_grads[submodule] = all_grads_out
            if save_grads:
                effect = t.mean(all_grads_out, dim=0) * deltas[submodule]
            else:
                effect = all_grads_out * deltas[submodule]
            effects[submodule] = effect

    del accumulated_in_grads, accumulated_out_grads
    gc.collect()
    for submodule in steer_submodules:
        if submodule.is_attn:
            qkv_in_grads[submodule] = {}
            for l in 'qkv':
                all_l_in_grads = t.cat(accumulated_qkv_in_grads[submodule][l], dim=0) if save_grads else accumulated_qkv_in_grads[submodule][l]
                qkv_in_grads[submodule][l] = all_l_in_grads
            all_head_grads = t.cat(accumulated_head_out_grads[submodule], dim=0) if save_grads else accumulated_head_out_grads[submodule]
            head_out_grads[submodule] = all_head_grads
            if save_grads:
                head_effect = t.mean(all_head_grads, dim=0) * head_deltas[submodule]
            else:
                head_effect = all_head_grads * head_deltas[submodule]
            head_effects[submodule] = head_effect
        
    del accumulated_head_out_grads, accumulated_qkv_in_grads
    gc.collect()

    return EffectOut(
        effects, 
        deltas, 
        in_grads, 
        out_grads, 
        total_effect,
        head_effects,
        head_deltas,
        head_out_grads,
        qkv_in_grads,
        None
    ), steer_submodules

def patching_effect(
        inputs,
        model,
        steer_vec,
        layer_idx,
        submodules: list[Submodule],
        metric_fn: Callable[[Envoy], t.Tensor],
        config=None,
        src_intervention_fn=None,
        dest_intervention_fn=None,
        neg_nodes=None,
        neg_edges=None,
):
    method, steps, trapezoidal, save_grads = config.method, config.n_steps, config.trapezoidal, config.save_grads
    if method == 'ig':
        return _pe_ig(inputs, model, steer_vec, layer_idx, submodules, metric_fn, src_intervention_fn, dest_intervention_fn, steps, trapezoidal, save_grads)
    elif method == 'exact':
        return _pe_exact(inputs, model, steer_vec, layer_idx, submodules, metric_fn, src_intervention_fn, dest_intervention_fn)
    # elif method == 'ap':
    #     return _pe_attrib(inputs, patch, model, submodules, metric_fn, metric_kwargs=metric_kwargs)
    else:
        raise ValueError(f"Unknown method {method}")

def compute_edge_simple(upstream_delta, downstream_grad, split_grads):
    """
    Simple ACDC-style edge: (∂L/∂v_in) × Δu
    """
    if split_grads:
        assert upstream_delta.shape == downstream_grad.shape[1:], f"{upstream_delta.shape}, {downstream_grad.shape}"
        avg_downstream_grad = downstream_grad.mean(0)
    else:
        assert upstream_delta.shape == downstream_grad.shape, f"{upstream_delta.shape}, {downstream_grad.shape}"
        avg_downstream_grad = downstream_grad
    edge_effect = upstream_delta * avg_downstream_grad
    
    return edge_effect, avg_downstream_grad

def obtain_ln_grads(input, model, submodules, downstream_grads, downstream_qkv_grads, vector, layer_idx, trapezoidal):
    # does not calculate grads for whole attention modules, only for qkv individual modules
    midstream_acts = []
    ln_grads = {}
    qkv_ln_grads = defaultdict(dict)
    upstream_submod = submodules[0]
    assert upstream_submod.name == f"resid_{layer_idx-1}"

    for i, down_submod in enumerate(submodules):
        if down_submod == upstream_submod:
            ln_grads[down_submod] = None
            continue
        downstream_in_grad = downstream_grads[down_submod]
        downstream_qkv_in_grad = downstream_qkv_grads[down_submod] if down_submod.is_attn else None
        num_steps, b, s, d = downstream_in_grad.shape
        
        accumulated_grads = []
        accumulated_qkv_grads = defaultdict(list)
        midstream_submods = submodules[1:i]
        for step in range(num_steps):
            alpha = (step+0.5) / (num_steps) if trapezoidal else step / num_steps
            with model.trace(input):
                # set steer
                res = model.model.layers[layer_idx-1].output.clone()
                res_steered = res + alpha*vector.to(model.device)
                model.model.layers[layer_idx-1].output = res_steered

                upstream_act = upstream_submod.get_out_activation()
                upstream_act.requires_grad_().retain_grad()

                for submodule in midstream_submods:
                    midmod = submodule.get_out_activation().detach()
                    submodule.set_out_activation(midmod) # breaks the graph
                    midstream_acts.append(midmod.save())

                if down_submod.is_attn:
                    q_act, k_act, v_act = down_submod.get_qkv_inputs_separate()
                    for l, l_act in zip('qkv', [q_act, k_act, v_act]):
                        score = (downstream_qkv_in_grad[l][step].to(model.device) * l_act).sum()
                        with score.backward(retain_graph=True):
                            grad_saved = upstream_act.grad.save()
                        
                        accumulated_qkv_grads[l].append(grad_saved.unsqueeze(0).detach())
                else:
                    downstream_act = down_submod.get_in_activation()
                    score = (downstream_in_grad[step].to(model.device) * downstream_act).sum()
                    with score.backward():
                        grad_saved = upstream_act.grad.save()
                    accumulated_grads.append(grad_saved.unsqueeze(0).detach())

        if down_submod.is_attn:
            for l in 'qkv':
                accumulated_l_grads = t.cat(accumulated_qkv_grads[l], dim=0)
                qkv_ln_grads[down_submod][l] = accumulated_l_grads.cpu()
        else:
            accumulated_grads = t.cat(accumulated_grads, dim=0)
            ln_grads[down_submod] = accumulated_grads.cpu()
    
    return ln_grads, qkv_ln_grads

def compute_edge_with_ln_slow(clean, model, upstream_submod, downstream_submod, upstream_delta, downstream_grad, vector, layer_idx, midstream_submods, trapezoidal):
    num_steps, b, s, d = downstream_grad.shape
    print(f"upstream: {upstream_submod.name}, downstream: {downstream_submod.name}")
    print(f"midstream: {[s.name for s in midstream_submods]}")
    accumulated_grads = []
    midstream_acts = []
    for step in range(num_steps):
        alpha = (step+0.5) / (num_steps) if trapezoidal else step / num_steps
        with model.trace(clean):
            # set steer
            res = model.model.layers[layer_idx-1].output.clone()
            res_steered = res + alpha*vector.to(model.device)
            model.model.layers[layer_idx-1].output = res_steered

            upstream_act = upstream_submod.get_out_activation()
            upstream_act.requires_grad_().retain_grad()

            for submodule in midstream_submods:
                midmod = submodule.get_out_activation().detach()
                submodule.set_out_activation(midmod) # breaks the graph
                midstream_acts.append(midmod.save())

            downstream_act = downstream_submod.get_in_activation()

            score = (downstream_grad[step].to(model.device) * downstream_act).sum()
            with score.backward():
                grad_saved = upstream_act.grad.save()
        accumulated_grads.append(grad_saved.unsqueeze(0).detach())
    
    accumulated_grads = t.cat(accumulated_grads, dim=0)
    avg_downstream_grad = accumulated_grads.mean(dim=0)
    edge_effect = upstream_delta.to(model.device) * avg_downstream_grad

    return edge_effect.cpu(), avg_downstream_grad.cpu()

    
if __name__ == "__main__":
    model_path = "google/gemma-2b-it"
    
    from nnsight import LanguageModel
    from .loading_utils import load_submodules

    model = LanguageModel(model_path, dtype=t.bfloat16, device_map="auto", dispatch=True)
    submodules = load_submodules(model, separate_by_type=True)
    steer_submodules, resid_submod = get_steer_submodules(submodules, 10)

    upstream_submod1 = Submodule("resid_9", model.model.layers[9])
    midstream_submods1 = [
        Submodule("attn_10", model.model.layers[10].self_attn),
        Submodule("mlp_10", model.model.layers[10].mlp),
        Submodule("attn_11", model.model.layers[11].self_attn),
        Submodule("mlp_11", model.model.layers[11].mlp),
        Submodule("attn_12", model.model.layers[12].self_attn),
        Submodule("mlp_12", model.model.layers[12].mlp),
    ]
    upstream_submod2 = Submodule("mlp_10", model.model.layers[10].mlp)
    midstream_submods2 = [
        Submodule("attn_11", model.model.layers[11].self_attn),
        Submodule("mlp_11", model.model.layers[11].mlp),
        Submodule("attn_12", model.model.layers[12].self_attn),
        Submodule("mlp_12", model.model.layers[12].mlp),
    ]
    downstream_submod = Submodule("attn_13", model.model.layers[13].self_attn)

    num_trials = 3
    for _ in range(num_trials):
        dim = model.config.hidden_size
        downstream_grads = {submod : t.randn((5,1,10,dim), dtype=t.bfloat16) for submod in steer_submodules}
        qkv_grads = {submod: {l: t.randn((5,1,10,dim)) for l in 'qkv'}
                     for submod in steer_submodules if submod.is_attn}

        if downstream_submod.is_attn:
            print('testing attn')
            gradsq = qkv_grads[downstream_submod]['q']
            gradsk = qkv_grads[downstream_submod]['k']
            deltaq = t.ones((1,10,dim))
            deltak = t.ones((1,10,dim))
        else:
            print('testing mlp module')
            gradsq = downstream_grads[downstream_submod]
            gradsk = downstream_grads[downstream_submod]
            deltaq = t.ones((1,10,dim))
            deltak = t.ones((1,10,dim))
        inputs = t.randint(10,1000, (1,10))

        vector = t.randn(2048, dtype=t.bfloat16, device=model.device)
        coeff = 1
        steer_vec = coeff*vector
        layer_idx = 10
        trapezoidal = True

        effects1q, grads1q = compute_edge_with_ln_slow(
            inputs, model, upstream_submod1, downstream_submod, deltaq, gradsq, steer_vec, layer_idx, midstream_submods1, trapezoidal
        )
        effects2q, grads2q = compute_edge_with_ln_slow(
            inputs, model, upstream_submod2, downstream_submod, deltaq, gradsq, steer_vec, layer_idx, midstream_submods2, trapezoidal
        )
        effects1k, grads1k = compute_edge_with_ln_slow(
            inputs, model, upstream_submod1, downstream_submod, deltak, gradsk, steer_vec, layer_idx, midstream_submods1, trapezoidal
        )
        effects2k, grads2k = compute_edge_with_ln_slow(
            inputs, model, upstream_submod2, downstream_submod, deltak, gradsk, steer_vec, layer_idx, midstream_submods2, trapezoidal
        )

        # print(f'Effects: {effects1.shape}')
        assert t.allclose(grads1q, grads2q)
        assert t.allclose(grads1k, grads2k)

        ln_grads, qkv_ln_grads = obtain_ln_grads(inputs, model, steer_submodules, downstream_grads, qkv_grads, steer_vec, layer_idx, trapezoidal)
        if downstream_submod.is_attn:
            effect_testq, test_gradq = compute_edge_simple(deltaq, qkv_ln_grads[downstream_submod]['q'])
            effect_testk, test_gradk = compute_edge_simple(deltak, qkv_ln_grads[downstream_submod]['k'])
        else:
            effect_testq, test_gradq = compute_edge_simple(deltaq, ln_grads[downstream_submod])
            effect_testk, test_gradk = compute_edge_simple(deltak, ln_grads[downstream_submod])

        assert t.allclose(grads1q, test_gradq.mean(dim=0))
        assert t.allclose(grads2q, test_gradq.mean(dim=0))
        assert t.allclose(effect_testq, effects1q)
        assert t.allclose(grads1k, test_gradk.mean(dim=0))
        assert t.allclose(grads2k, test_gradk.mean(dim=0))
        assert t.allclose(effect_testk, effects1k)

        print(f'last submodule: {steer_submodules[-1].name}')
        print(ln_grads[steer_submodules[-1]].shape)
        print(downstream_grads[steer_submodules[-1]].shape)
        assert not t.allclose(ln_grads[steer_submodules[-1]], downstream_grads[steer_submodules[-1]])
        
    print('passed')