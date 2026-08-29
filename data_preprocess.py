"""
Checking the KL Divergence for a specific steering vector on a specific dataset.
Then save the completions
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional
import pandas as pd
import json
import matplotlib.pyplot as plt
from copy import deepcopy
import math
from tqdm import tqdm
import argparse
from config_act_patch import dim_layer_pos_dict, dim_layer_map

from transformers import AutoModelForCausalLM, AutoTokenizer
from mat_hooks import add_vec_hooks
from mat_hooks import get_activation_addition_input_pre_hook
from utils.gen_utils import load_model_and_tokenizer, set_seeds
from utils.data_utils import load_test_prompt, load_refusal_completions, load_steering_vector
from utils.chat_utils import get_chat_template_fn

set_seeds()

def visualize_token_heatmap(tokens, heatmap_values, tokenizer, prompt_len=0, cmap='RdYlGn',
                           fig_width=12, font_size=9):
    """
    Visualize tokens with heatmap highlighting, with automatic line wrapping.

    Uses a two-pass approach: first measures actual rendered text widths, then
    computes layout and builds a figure sized to fit the content exactly.

    Args:
        tokens: list/tensor of token IDs
        heatmap_values: list/array of numerical values (same length as tokens)
        tokenizer: tokenizer to decode tokens
        prompt_len: number of prompt tokens; these are rendered in gray without color
        cmap: matplotlib colormap name
        fig_width: figure width in inches
        font_size: font size in points
    """
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors import Normalize

    token_list = tokens.tolist() if hasattr(tokens, 'tolist') else list(tokens)

    def clean(tok_id):
        s = tokenizer.decode([tok_id])
        return s.replace('\n', '↵').replace('\r', '').replace('\t', '⇥') or '·'

    texts = [clean(t) for t in token_list]

    norm_values = np.array(heatmap_values, dtype=float)
    vmin, vmax = norm_values.min(), norm_values.max()
    norm_scaled = (norm_values - vmin) / (vmax - vmin + 1e-10)
    colormap = plt.cm.get_cmap(cmap)

    def get_color(i):
        if i < prompt_len:
            return (0.72, 0.72, 0.72), 0.6
        val = norm_scaled[i - 1] if i > 0 else 0.0
        return tuple(colormap(val)[:3]), 0.85

    # ------------------------------------------------------------------
    # Pass 1: measure actual text pixel widths via a temporary figure.
    # Setting up the axes so 1 data unit == 1 inch lets us use measured
    # inch widths directly as data-coordinate widths in the final figure.
    # ------------------------------------------------------------------
    dpi = plt.rcParams['figure.dpi']

    fig_tmp, ax_tmp = plt.subplots(figsize=(fig_width, 2), dpi=dpi)
    ax_tmp.axis('off')
    tmp_objs = [ax_tmp.text(0, 0, t, fontsize=font_size, family='monospace')
                for t in texts]
    ref_obj  = ax_tmp.text(0, 0, 'Ag', fontsize=font_size, family='monospace')
    fig_tmp.canvas.draw()
    rend = fig_tmp.canvas.get_renderer()

    widths_in = [t.get_window_extent(rend).width  / dpi for t in tmp_objs]
    ref_h_in  = ref_obj.get_window_extent(rend).height / dpi
    plt.close(fig_tmp)

    # ------------------------------------------------------------------
    # Pass 2: compute flow layout in inches.
    # bbox pad=0.2 adds 0.2*font_size pts = (0.2*font_size/72) in per side.
    # ------------------------------------------------------------------
    margin_in = 0.2
    gap_in    = 0.05              # horizontal gap between boxes
    pad_in    = 0.2 * font_size / 72.0   # bbox padding per side (matches pad=0.2)
    line_h_in = (ref_h_in + 2 * pad_in) * 1.8  # line spacing
    avail_w   = fig_width - 2 * margin_in

    positions = []   # (x_text_anchor, line_index)
    x_pos, line_idx = 0.0, 0
    for w in widths_in:
        box_w = w + 2 * pad_in
        if x_pos + box_w > avail_w and x_pos > 0.0:
            x_pos    = 0.0
            line_idx += 1
        positions.append((x_pos + pad_in, line_idx))
        x_pos += box_w + gap_in

    n_lines = line_idx + 1

    # ------------------------------------------------------------------
    # Build final figure with height derived from the computed layout.
    # Axes are sized so that 1 data unit == 1 inch in both x and y,
    # making the measured inch-widths directly usable as data coordinates.
    # ------------------------------------------------------------------
    top_pad_in = 0.1
    cbar_h_in  = 0.45
    bot_pad_in = 0.5
    main_h_in  = n_lines * line_h_in + top_pad_in
    fig_h      = main_h_in + cbar_h_in + bot_pad_in

    fig = plt.figure(figsize=(fig_width, fig_h), dpi=dpi)

    # Main axes: left/right margins of margin_in, sits above colorbar area
    ax = fig.add_axes([
        margin_in / fig_width,
        (cbar_h_in + bot_pad_in) / fig_h,
        avail_w / fig_width,
        main_h_in / fig_h,
    ])
    ax.set_xlim(0, avail_w)
    ax.set_ylim(-main_h_in + top_pad_in, top_pad_in)
    ax.axis('off')

    for i, (text, (x_anchor, li)) in enumerate(zip(texts, positions)):
        fc, alpha = get_color(i)
        y_anchor  = -(li * line_h_in) - pad_in   # top of text, pad below line top
        ax.text(x_anchor, y_anchor, text,
                fontsize=font_size, family='monospace',
                ha='left', va='top',
                bbox=dict(boxstyle='square,pad=0.2', facecolor=fc,
                          edgecolor='none', alpha=alpha))

    # Colorbar
    cax = fig.add_axes([0.1, bot_pad_in / fig_h, 0.8, cbar_h_in * 0.55 / fig_h])
    norm = Normalize(vmin=vmin, vmax=vmax)
    cb = ColorbarBase(cax, cmap=colormap, norm=norm, orientation='horizontal')
    cb.set_label('KL Divergence', labelpad=6)

    return fig, ax

def examine_top_k(logits, position, tokenizer):
    topk_tok = torch.topk(logits[position], 5)
    for i in range(len(topk_tok[1])):
        tok = topk_tok[1][i]
        val = topk_tok[0][i]
        print(f"token {i}: {tokenizer.decode(tok)} with value {val}")
    return topk_tok

def fwd_with_hooks(model, inputs, fwd_hooks=[], fwd_pre_hooks=[]):
    if type(inputs) == dict:
        for k in inputs.keys(): inputs[k] = inputs[k].to(model.device)
    else:
        inputs = inputs.to(model.device)

    # run it again but with steering vectors
    handles = []
    for module, hook in fwd_pre_hooks:
        handles.append(module.register_forward_pre_hook(hook))
    for module, hook in fwd_hooks:
        handles.append(module.register_forward_hook(hook))

    with torch.no_grad():
        out = model(**inputs)
    
    for h in handles:
        h.remove()
    
    return out

def get_steer_hooks(model, layer_idx, vector, coeff):
    fwd_pre_hooks = [(model.model.layers[layer_idx], get_activation_addition_input_pre_hook(vector=vector, coeff=coeff))]
    fwd_hooks = []
    return fwd_hooks, fwd_pre_hooks

def format_input(tokenizer, model_path, contents: list[list[dict]]):
    chat_template_fn = get_chat_template_fn(model_path)
    strings = [chat_template_fn(convo) for convo in contents]
    inputs = tokenizer(strings, return_tensors='pt', padding=True, add_special_tokens=False)
    return inputs, strings

def gen_with_hooks(model, inputs, fwd_pre_hooks, fwd_hooks):
    # run it again but with steering vectors
    handles = []
    for module, hook in fwd_pre_hooks:
        handles.append(module.register_forward_pre_hook(hook))
    for module, hook in fwd_hooks:
        handles.append(module.register_forward_hook(hook))

    inputs = inputs.to(model.device)
    with torch.no_grad():
        output_steer = model.generate(
            **inputs, 
            max_new_tokens=100, 
            do_sample=False,
            temperature=1)
    
    for h in handles:
        h.remove()
    
    return output_steer



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='google/gemma-2-2b-it')
    parser.add_argument('--layer', type=int)
    parser.add_argument('--concept', type=str, default="refusal")
    parser.add_argument('--harm_flag', action="store_true", default=False)
    parser.add_argument('--steer_flag', action="store_true", default=False)
    parser.add_argument('--learn_type', type=str, default='dim', choices=['dim', 'ntp', 'reps', 'ortho'])
    parser.add_argument('--learn_path', type=str, default=None)
    
    args = parser.parse_args()

    # load steering vector and model
    concept = args.concept
    model_path = args.model_path
    steer_flag = args.steer_flag
    layer = dim_layer_map[model_path]
    layer, pos = dim_layer_pos_dict[model_path][layer]
    concept_params = {
        'refusal': {'harm_flag': args.harm_flag}
    }[concept]

    learned = False if args.learn_type == 'dim' else True
    if learned:
        suffix = f"{('harmful' if args.harm_flag else 'harmless')}q_{('steered' if steer_flag else 'base')}_{args.learn_path}"
    else:
        suffix = f"{('harmful' if args.harm_flag else 'harmless')}q_{('steered' if steer_flag else 'base')}_layer{layer}_pos{pos}"

    vector, coeff, layer_idx = load_steering_vector(
        concept=args.concept, 
        model_path=args.model_path, 
        concept_params=concept_params, 
        pos=pos, 
        layer_idx=layer, 
        learned=learned,
        vector_base_path=None if args.learn_path == 'None' else args.learn_path,
    )

    print(f'loaded {args.learn_type} vector shape {vector.shape} with coeff {coeff} at layer {layer_idx}')
    model, tokenizer = load_model_and_tokenizer(model_path)

    # load completions
    # all_completions = load_test_prompt()
    if concept == 'refusal':
        all_completions = load_refusal_completions(
            model_path=model_path, 
            steer_flag=steer_flag, 
            concept_params=concept_params, 
            learned=learned,
            pos=pos, 
            layer=layer, 
            vector_base_path=args.learn_path,
        )
        print(f'Loaded {len(all_completions)} samples')
    else:
        raise ValueError("only supports refusal right now")

    # divide prompts based on kl divergence thresholds
    width = 2
    thresholds = list(range(width,40,width))
    clusters = {f"[{float(s - width)},{float(s)})": [] for s in thresholds}

    all_data = []

    print('processing completions...')
    for prompt_contents in tqdm(all_completions):
        prompt_inputs, _ = format_input(tokenizer, model_path, [[prompt_contents[0]]])

        prompt_lens = [len(prompt) for prompt in prompt_inputs['input_ids']]

        convo_inputs, convo_strings = format_input(tokenizer, model_path, [prompt_contents])

        fwd_hooks, fwd_pre_hooks = get_steer_hooks(model, layer_idx, vector, coeff)
        outputs_steer = fwd_with_hooks(model, convo_inputs, fwd_hooks=fwd_hooks, fwd_pre_hooks=fwd_pre_hooks)
        outputs_base = fwd_with_hooks(model, convo_inputs, fwd_hooks=[], fwd_pre_hooks=[])

        logits_base = outputs_base.logits.detach().float() # float32 for high precision
        logits_steer = outputs_steer.logits.detach().float()

        base_ys = torch.argmax(logits_base, dim=-1)
        steer_ys = torch.argmax(logits_steer, dim=-1)
        mismatches = (base_ys != steer_ys).cpu().numpy()

        if steer_flag:
            # using misaligned sample: checking divergence of no steer probability against steer probability
            log_probs_input = F.log_softmax(logits_base, dim=-1)
            probs_target = F.softmax(logits_steer, dim=-1)
        else:
            # using aligned sample: checking divergence of steer probability against no steer probability
            log_probs_input = F.log_softmax(logits_steer, dim=-1)
            probs_target = F.softmax(logits_base, dim=-1)

        kl_divs_batch = F.kl_div(log_probs_input, probs_target, reduction='none').sum(dim=-1)
        kl_divs_batch = torch.clamp_min(kl_divs_batch, 0)
        kl_divs_batch = kl_divs_batch.cpu().numpy()
        assert kl_divs_batch.shape == convo_inputs['input_ids'].shape

        for sample, convo_string, kl_divs, prompt_len, mismatch in zip(convo_inputs['input_ids'], convo_strings, kl_divs_batch, prompt_lens, mismatches):
            for i in range(prompt_len-1, len(sample)):
                kl_div = kl_divs[i]
                c = kl_div // width
                clusters[f"[{c*width},{(c+1)*width})"].append(tokenizer.decode(sample[:i+1]))
            assert (tokenizer(convo_string, return_tensors="pt", add_special_tokens=False)['input_ids'] == sample.to("cpu")).all(), print(tokenizer(convo_string, return_tensors="pt")['input_ids'], '\n', sample)
            assert convo_string == tokenizer.decode(sample), print(convo_string, '\n', tokenizer.decode(sample))
            all_data.append({
                'strings': convo_string,
                'tokens': sample.to('cpu').numpy(),
                'prompt_length': prompt_len,
                'kl_divergence': kl_divs,
                'mismatch': mismatch,
                'concept': concept,
                'concept_params': concept_params,
                'steer_flag': steer_flag,
            })
    
    # save pandas dataframe with queries and kl_divergence
    df = pd.DataFrame(all_data)
    save_dir = f"data/kl_divs/{concept}/{model_path.split('/')[-1]}"
    os.makedirs(save_dir, exist_ok=True)

    df.to_parquet(os.path.join(save_dir, f"kl_analysis_{suffix}.parquet"), index=False)


    # analyze histogram
    num_per_cluster = [len(v) for k, v in clusters.items()]
    print(num_per_cluster)
    plt.bar(np.arange(len(num_per_cluster))*2+1, num_per_cluster, width=1)
    plt.xticks(np.arange(len(num_per_cluster))*2+1, clusters.keys())
    plt.yscale('log')
    plt.savefig(os.path.join(save_dir, f"histogram_{concept}_{suffix}.png"))
    print("created histogram")

    # visualize one sample
    fig, axs = visualize_token_heatmap(convo_inputs['input_ids'][0], kl_divs, tokenizer, prompt_len=prompt_lens[0])
    if steer_flag:
        plt.savefig(os.path.join(save_dir, f'check_heat_{suffix}.png'))
    else:
        plt.savefig(os.path.join(save_dir, f'check_heat_{suffix}.png'))
    print("saved heatmap")
