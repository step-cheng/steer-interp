import contextlib
from typing import List, Tuple, Callable
from jaxtyping import Float
from torch import Tensor
import torch
from functools import partial

@contextlib.contextmanager
def add_hooks(
    module_forward_pre_hooks: List[Tuple[torch.nn.Module, Callable]],
    module_forward_hooks: List[Tuple[torch.nn.Module, Callable]],
    **kwargs
):
    """
    Context manager for temporarily adding forward hooks to a model.

    Parameters
    ----------
    module_forward_pre_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward pre hook on the module
    module_forward_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward hook on the module
    """
    try:
        handles = []
        for module, hook in module_forward_pre_hooks:
            partial_hook = partial(hook, **kwargs)
            handles.append(module.register_forward_pre_hook(partial_hook))
        for module, hook in module_forward_hooks:
            partial_hook = partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
        yield
    finally:
        for h in handles:
            h.remove()

def _act_add_hook(module, input, output, steer_vec, **kwargs):
    """
    Should be a post hook
    """
    device = output[0].device
    modified = output[0][:,:,:] + steer_vec.reshape(1,1,-1).to(device)
    return (modified,) + output[1:]

def _act_subtract_hook(module, input, output, steer_vec, **kwargs):
    """
    Should be a post hook
    """
    device = output[0].device
    modified = output[0][:,:,:] - steer_vec.reshape(1,1,-1).to(device)
    return (modified,) + output[1:]

def _proj_add_hook(module, input, output, vec, scale, **kwargs):
    """
    Should be a post hook
    """
    # steering_vector must be unit-normed: r̂
    # Project out r̂ from all positions in the batch and sequence
    r_hat = vec / vec.norm()  # (d_model,)
    projection = output[0] @ r_hat  # shape: (batch, seq)
    output[0][:, :, :] += scale * projection.unsqueeze(-1) * r_hat

def _store_iics(module, input, output, vec, layer, pre_ln_activations, **kwargs):
    """
    Should be a pre hook
    """
    
def _store_pre_ln_hook(module, input, pre_ln_activations: dict, layer_idx: int):
    """
    Store activation stats before the layer norm
    """
    if isinstance(ins, tuple) and len(ins) > 0:
        ins = ins[0]
    pre_ln_activations[layer_idx] = ins

def _patch_hook(module, input, output, clean_acts, layer_idx, mod):
    """
    Patch in activation at a module.
    """
    source_act = clean_acts[f'layer_{layer_idx}_{mod}']
    if isinstance(output, tuple):
        assert output[0].shape == source_act.shape
        return (source_act,) + output[1:]
    assert output.shape == source_act.shape
    return source_act
    
def _use_clean_attn_hook(module, input, clean_acts, layer_idx):
    """
    Uses the clean activation as input to attn module. 
    Then, run the module, and store/add this in the running activation object. 
    Should be pre-hook
    """
    clean_in = clean_acts[f'layer_{layer_idx}_pre_attn']
    clean_out, _ = module(clean_in)
    clean_acts['running'] += clean_out
    clean_acts[f'layer_{layer_idx}_attn'] = clean_out

def _use_clean_mlp_hook(module, input, clean_acts, layer_idx):
    """
    Uses the clean activation as input to mlp module. 
    Then, run the module, and store/add this in the running activation object. 
    Should be pre-hook
    """
    clean_in = clean_acts[f'layer_{layer_idx}_pre_mlp']
    clean_out = module(clean_in)
    clean_acts['running'] += clean_out
    clean_acts[f'layer_{layer_idx}_mlp'] = clean_out

def _use_clean_norm_hook(module, input, clean_acts, layer_idx, mod):
    """
    Uses the clean activation as input to layernorm module. 
    Then, run the module, and store/add this in the running activation object. 
    Should be pre-hook
    """
    clean_in = clean_acts['running']
    clean_out = module(clean_in)
    if mod == "attn":
        clean_acts[f'layer_{layer_idx}_pre_attn'] = clean_out
    elif mod == "mlp":
        clean_acts[f'layer_{layer_idx}_pre_mlp'] = clean_out

def _obtain_clean_act_hook(module, input, output, clean_acts):
    """
    Store clean activation before steering is applied.
    Should be registered BEFORE the steering hook.
    """
    if isinstance(output, tuple):
        clean_acts['running'] = output[0].clone()
    else:
        clean_acts['running'] = output.clone()
    return output

def get_activation_addition_input_pre_hook(vector: Float[Tensor, "d_model"], coeff: Float[Tensor, ""]):
    def hook_fn(module, input):
        nonlocal vector

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        vector = vector.to(activation)
        activation = activation + coeff * vector

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn

# ---------------------------------

def add_vec_hooks(model, layer_idx, coeff, vector):
    """
    Baseline: add vec at desired layer_idx
    """
    layers = model.layers
    forward_hooks = []
    forward_pre_hooks = []
    forward_hooks.append((layers[layer_idx], partial(_act_add_hook, steer_vec=coeff*vector)))
    
    return forward_hooks, forward_pre_hooks

def add_then_subtract_vec_hooks(model, layer_idx, coeff, vector, d):
    """
    Exp 1: subtract original vec after d layers
    """
    layers = model.layers
    forward_hooks = []
    forward_pre_hooks = []
    forward_hooks.append((layers[layer_idx], partial(_act_add_hook, steer_vec=coeff*vector)))
    forward_hooks.append((layers[layer_idx+d], partial(_act_subtract_hook, steer_vec=coeff*vector)))
    
    return forward_hooks, forward_pre_hooks

def test_hooks(model, layer_idx, coeff, vector, d):
    """
    Sanity check exp1
    """
    layers = model.layers
    forward_hooks = []
    forward_pre_hooks = []
    forward_hooks.append((layers[layer_idx], partial(_act_add_hook, steer_vec=coeff*vector)))
    forward_hooks.append((layers[layer_idx+d], partial(_act_subtract_hook, steer_vec=coeff*vector)))
    forward_hooks.append((layers[layer_idx+d], partial(_act_add_hook, steer_vec=coeff*vector)))
    
    return forward_hooks, forward_pre_hooks

def delay_add_vec_hooks(model, layer_idx, coeff, vector, d):
    """
    Exp 2: add original vec after d layers
    """
    layers = model.layers
    forward_hooks = []
    forward_pre_hooks = []
    forward_hooks.append((layers[layer_idx+d], partial(_act_add_hook, steer_vec=coeff*vector)))

    return forward_hooks, forward_pre_hooks

def patching_hooks(model, layer_idx, coeff, vector, locations: list[Tuple[int, str]]):
    """
    Exp 3: Add original vec and patch clean acts at each location
    1. add post hook to layer_idx to capture clean acts
    2. add post hook to layer_idx to add steer vector
    3. add clean act calculation hooks to each attn, mlp and layernorm
    4. add patch hooks to desired places in locations
    """
    forward_hooks = []
    forward_pre_hooks = []
    clean_acts = {}

    layers = model.layers
    forward_hooks.append((layers[layer_idx], partial(_obtain_clean_act_hook, clean_acts=clean_acts)))
    forward_hooks.append((layers[layer_idx], partial(_act_add_hook, steer_vec=coeff*vector)))

    for i, layer in enumerate(layers[layer_idx+1:]):
        if not hasattr(layer, "self_attn"):
            raise RuntimeError("Missing self_attn")
        if not hasattr(layer, "input_layernorm"):
            raise RuntimeError("Missing input_layernorm")
        if not hasattr(layer, "post_attention_layernorm"):
            raise RuntimeError("Missing post_attention_layernorm")
        if not hasattr(layer, "mlp"):
            raise RuntimeError("Missing mlp")
        idx = layer_idx + 1 + i
        in_ln, attn, post_ln, mlp = layer.input_layernorm, layer.self_attn, layer.post_attention_layernorm, layer.mlp
        forward_pre_hooks.append((in_ln, partial(_use_clean_norm_hook, clean_acts=clean_acts, layer_idx=idx, mod="attn")))
        forward_pre_hooks.append((attn, partial(_use_clean_attn_hook, clean_acts=clean_acts, layer_idx=idx)))
        forward_pre_hooks.append((post_ln, partial(_use_clean_norm_hook, clean_acts=clean_acts, layer_idx=idx, mod="mlp")))
        forward_pre_hooks.append((mlp, partial(_use_clean_mlp_hook, clean_acts=clean_acts, layer_idx=idx)))

    for idx, mod in locations:
        layer = layers[idx]
        if mod == "attn":
            module = layer.self_attn
        elif mod == "mlp":
            module = layer.mlp
        forward_hooks.append((module, partial(_patch_hook,
                                            clean_acts=clean_acts,
                                            layer_idx=idx,
                                            mod=mod)))

    return forward_hooks, forward_pre_hooks, clean_acts
    