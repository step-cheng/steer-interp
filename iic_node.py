import pandas as pd
import seaborn as sns
import matplotlib.font_manager as fm
import torch as t
from patching.loading_utils import Submodule
from collections import defaultdict
from config_act_patch import dim_layer_pos_dict
import json

class IICNode:
    """
    This only works from the steer resid to the immediate value, and no other way.
    """
    def __init__(self, model, layer_idx):
        self.config = model.config
        pre_ln = model.model.layers[layer_idx].input_layernorm
        attn = model.model.layers[layer_idx].self_attn

        if model.config._name_or_path.startswith('google/gemma-2-'):
            post_ln = model.model.layers[layer_idx].post_attention_layernorm
        else:
            post_ln = None

        self.device = attn.device

        self.d_model = attn.config.hidden_size
        self.num_kv_groups = attn.config.num_attention_heads // attn.config.num_key_value_heads
        self.num_kv_heads = attn.config.num_key_value_heads
        self.head_dim = getattr(attn, 'head_dim', attn.config.hidden_size // attn.config.num_attention_heads)

        self.W_V = attn.v_proj.weight.to(self.device)   # num_kv_heads*head_dim, hidden_size
        self.b_V = attn.v_proj.bias if hasattr(attn.v_proj, 'bias') else None  # num_kv_heads*head_dim
        self.W_O = attn.o_proj.weight.to(self.device)   # hidden_size, num_attn_heads * head_dim
        self.b_O = attn.o_proj.bias if hasattr(attn.o_proj, 'bias') else None    # hidden_size
        self.pre_ln_weight = pre_ln.weight.to(self.device)

        if hasattr(pre_ln, "eps"):
            self.pre_ln_eps = pre_ln.eps
        elif hasattr(pre_ln, "variance_epsilon"):
            self.pre_ln_eps = pre_ln.variance_epsilon
        self.post_ln_weight = None if post_ln is None else post_ln.weight.to(self.device)
        self.post_ln_eps = None if post_ln is None else post_ln.eps

    def get_pre_rms_norm_stat(self, x):
        return t.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.pre_ln_eps)

    def get_post_rms_norm_stat(self, x):
        return t.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.post_ln_eps)
    
    def pre_ln_norm_v(self, raw_in_v, ln_stats=1):
        in_v = raw_in_v.float() * ln_stats
        if self.config._name_or_path.startswith('google/gemma-2-'):
            in_v = in_v * (1.0 + self.pre_ln_weight.float())
        elif self.config._name_or_path.startswith('meta-llama/Llama-3.2-'):
            in_v = in_v * self.pre_ln_weight.float()
        return in_v.type_as(raw_in_v)

    def post_ln_norm_v(self, raw_in_v, ln_stats=1):
        if self.post_ln_weight is not None:
            in_v = raw_in_v.float() * ln_stats
            if self.config._name_or_path.startswith('google/gemma-2-'):
                in_v = in_v * (1.0 + self.post_ln_weight.float())
            elif self.config._name_or_path.startswith('meta-llama/Llama-3.2-'):
                in_v = in_v * self.post_ln_weight.float()
            return in_v.type_as(raw_in_v)
        else:
            return raw_in_v

    def get_out_vector(self, norm_in_v):
        VW_V = t.matmul(norm_in_v, self.W_V.T)
        if self.b_V is not None:
            VW_V = VW_V + self.b_V
        *leading_dims, _ = VW_V.shape
        VW_V = VW_V.view(*leading_dims, self.num_kv_heads, self.head_dim)
        
        # Repeat along head dimension (dim=-2), not element-wise
        VW_V = VW_V.repeat_interleave(self.num_kv_groups, dim=-2)
        W_O_heads = self.W_O.reshape(self.d_model, -1, self.head_dim)
        VW_OV = t.einsum('nh,dnh->nd', VW_V, W_O_heads)
        
        if self.b_O is not None:
            VW_OV = VW_OV + self.b_O
        
        return VW_OV
    
    def __str__(self):
        return f"LN: epsilon {self.ln_eps}, weight {self.ln_weight.shape}. Attn: W_V {self.W_V.shape}, b_V {'None' if not self.b_V else self.b_V.shape}, W_O {self.W_O.shape}, b_O {'None' if not self.b_O else self.b_O.shape}, num_kv_groups {self.num_kv_groups}"


def logit_lens(x):
    with t.no_grad():
        final_ln = model.model.norm
        x = x.to(final_ln.device)
        inp = final_ln(x)
        lm_head = model.lm_head
        logits = lm_head(inp)

    # cosine sim
    W = lm_head.weight  # (vocab_size, d_model)
    inp_u = inp / inp.norm()
    W_u = W / W.norm(dim=-1, keepdim=True)
    cos_sims = (W_u @ inp_u).float()

    return logits, cos_sims

def get_topk(out):
    vals, inds = t.topk(out, k=20, dim=-1)
    return inds, vals

def plot(iic_vectors, steer_vec, heads_to_plot, plot_flag=False, save_path=None, m='logit'):
    print('PLOTTING')
    
    to_plot = {}
    
    emoji_replacements = {
        '🙅': '[x_arms]',
        '❌': '[x]',
        '多': '[many]',
        '对不起': '[sorry]',
        ' 👎': '[thumbs_down]',
        '禁止': 'prohibit',
        '关停': '[shut down]',
        '不应该': '[should not]',
        '根本不': '[not at all]',
        '严禁': '[strictly prohibit]',
        '违背': '[violate]',
        '但这': '[but this]',
        '这不是': '[this is not]',
        '但它': '[but it]',
        '非法': '[illegal]',
    }

    for k, v in heads_to_plot.items():
        if k == "Sum":
            all_iic = t.cat([out_vectors for layer_idx, out_vectors in iic_vectors.items()], dim=0)
            vector = t.sum(all_iic, dim=0)
            key = "SUM"
        elif k == "sv":
            vector = steer_vec
            key = "SV"
        else:
            layer = int(k.split('.')[0])
            head = int(k.split('.')[1])
            print(iic_vectors.keys(), layer)
            if head < 0:
                vector = iic_vectors[layer][abs(head)] * -1
            else:
                vector = iic_vectors[layer][abs(head)]
            key = f"{'-' if head < 0 else ''}L{layer}H{abs(head)}"

        logits, sims = logit_lens(vector)
        inds, vals = get_topk(logits)
        toks_to_plot = [emoji_replacements.get(tokenizer.decode(ind), tokenizer.decode(ind)) for ind in inds.flatten()[v]]
        
        if m == 'logit':
            heats_to_plot = logits[inds].flatten()[v].tolist()
        elif m == 'sim':
            heats_to_plot = sims[inds].flatten()[v].tolist()
        to_plot[key] = toks_to_plot, heats_to_plot


    def is_emoji(char):
        """Check if character is an emoji"""
        return ord(char) > 0x1F000
    # Create the heatmap visualization
    def plot_logit_lens_seaborn(to_plot, save_path=save_path):
        # Prepare DataFrame
        rows = []
        for layer_head, (tokens, logits) in to_plot.items():
            for rank, (token, logit) in enumerate(zip(tokens, logits)):
                rows.append({
                    'Layer.Head': layer_head,
                    'Rank': f'k={rank+1}',
                    'Token': token,
                    'Logit': logit
                })
        
        df = pd.DataFrame(rows)
        
        # Pivot for heatmap (reindex to preserve original insertion order)
        original_order = list(to_plot.keys())
        pivot = df.pivot(index='Layer.Head', columns='Rank', values='Logit').reindex(original_order)
        tokens_pivot = df.pivot(index='Layer.Head', columns='Rank', values='Token').reindex(original_order)
        
        # Create plot with separate colorbar axes for equal column spacing
        fig, (ax, cbar_ax) = plt.subplots(1, 2, figsize=(8, 6),
                                          gridspec_kw={'width_ratios': [1, 0.04]})

        # Heatmap
        hm = sns.heatmap(pivot, annot=tokens_pivot.values, fmt='',
                    cmap='YlOrRd',
                    linewidths=2, linecolor='white',
                    annot_kws={'fontsize': 10, 'fontweight': 'bold'},
                    ax=ax, cbar_ax=cbar_ax)

        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0,
                           fontweight='bold', fontsize=10)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right',
                           fontweight='bold', fontsize=10)


        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
        return fig

    if plot_flag and save_path is not None:    
        fig = plot_logit_lens_seaborn(to_plot)

        with open(f"notebooks/svv_{model_path.split('/')[-1]}_{args.learn_type}_{m}.json", 'w') as f:
            json.dump(to_plot, f)


def inspect_iic(model, tokenizer, steer_vec, steer_layer_idx, offset: int=0, offset_head: int=0, view_heads: bool = True, view_head_dict: dict = None, look_at_negatives: bool =True):
    num_heads, num_hidden_layers = model.config.num_attention_heads, model.config.num_hidden_layers
    
    print(f'set steering vector with positive steer coeff')
    print()

    if offset > 0:
        iic_attn = IICNode(model, steer_layer_idx+offset-1)
        steer_vec = steer_vec.to(iic_attn.device)
        norm_steer_vec = iic_attn.pre_ln_norm_v(steer_vec)
        out_vector = iic_attn.get_out_vector(norm_steer_vec)
        steer_vec = iic_attn.post_ln_norm_v(out_vector)[offset_head]
    
    iic_vectors = {}
    for layer_idx in range(steer_layer_idx+offset, num_hidden_layers):
        print(f'layer idx {layer_idx}')

        iic_attn = IICNode(model, layer_idx)

        steer_vec = steer_vec.to(iic_attn.device)
        norm_steer_vec = iic_attn.pre_ln_norm_v(steer_vec)
        out_vector = iic_attn.get_out_vector(norm_steer_vec)
        out_vector = iic_attn.post_ln_norm_v(out_vector)

        iic_vectors[layer_idx] = out_vector

        if view_heads:
            if layer_idx not in view_head_dict:
                continue
            hoi = view_head_dict[layer_idx]
            for h in hoi:
                print(f'head {h}')
                if h < 0 and look_at_negatives:
                    out, _ = logit_lens(-out_vector[abs(h)])
                else:
                    out, _ = logit_lens(out_vector[h])
                # out = logit_lens(out_vector[abs(h)])
                inds, _ = get_topk(out)

                print([tokenizer.decode(ind) for ind in inds])
        print(f'summed')
        out, _ = logit_lens(out_vector.sum(dim=0))
        inds, _ = get_topk(out)

        print("|".join([tokenizer.decode(ind) for ind in inds]))

        print()
    
    all_iic_vectors = t.cat([out_vectors for _, out_vectors in iic_vectors.items()], dim=0)
    print(all_iic_vectors.shape)
    print(t.norm(all_iic_vectors, dim=-1))

    print(f"summed approximate contributions of iic")
    sum_iic_vector = t.sum(all_iic_vectors, dim=0)
    logits, _ = logit_lens(sum_iic_vector)
    inds, _ = get_topk(logits)
    print([tokenizer.decode(ind) for ind in inds])

    print('steer vec logit lens')
    logits, _ = logit_lens(steer_vec)
    inds, _ = get_topk(logits)
    print([tokenizer.decode(ind) for ind in inds])

    return iic_vectors

def convert_top_head_list_to_dict(head_list, sign_list=None):
    d = defaultdict(list)
    if sign_list is None:
        sign_list = [1] * len(head_list)
    for h, sign in zip(head_list, sign_list):
        splits = h.split('_')
        layer = splits[1]
        head = splits[2][1:]
        d[int(layer)].append(int(head)*int(sign))
    return d

if __name__ == '__main__':
    from nnsight import LanguageModel
    from utils.data_utils import load_steering_vector, load_refusal_completions
    from data_preprocess import format_input
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from patching.loading_utils import load_submodules, Submodule
    import matplotlib.pyplot as plt
    import torch.nn.functional as F
    import numpy as np
    import seaborn as sns
    import argparse
    from config_act_patch import dim_layer_map

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--harm_flag", action="store_true", default=False)
    parser.add_argument("--learn_type", type=str, choices=['dim', 'ntp', 'reps', 'ortho'])
    parser.add_argument("--learn_path", type=str, default=None)
    args = parser.parse_args()

    model_path = args.model_path
    layer = dim_layer_map[model_path]
    print('loading model')
    model = LanguageModel(model_path, dispatch=True, dtype=t.float32, device_map='auto')
    print(model.config.head_dim)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    num_heads = model.config.num_attention_heads

    harm_flag = args.harm_flag

    if args.learn_type == 'dim':
        learned = False
        layer, pos = dim_layer_pos_dict[model_path][layer]
    else:
        learned = True
        layer, pos = layer, None
    vector, coeff, steer_layer_idx = load_steering_vector(
        concept="refusal", 
        model_path=model_path, 
        concept_params={'harm_flag': harm_flag}, 
        pos=pos, 
        layer_idx=layer,
        learned=learned,
        vector_base_path=args.learn_path
    )
    print(f'loaded vector {vector.shape} with coeff {coeff} at layer {layer}, norm {vector.norm()}')

    steer_vec = vector.float() * coeff / vector.norm().item()
    print(steer_vec.norm())

    # view_head_dict_pos = {
    #     12: [9,22],
    #     13: [18],
    #     14: [3, 19],
    #     15: [15]
    # }
    # view_head_dict_pos = {
    #     12: [9,22],
    #     13: [18],
    #     14: [3, 19],
    #     15: [15]
    # }
    view_head_dict_pos = convert_top_head_list_to_dict(
        ['attn_22_h6', 'attn_20_h9', 'attn_23_h1', 'attn_22_h16', 'attn_22_h4', 'attn_20_h19', 'attn_20_h10', 'attn_22_h7', 'attn_22_h5', 'attn_20_h8', 'attn_21_h12', 'attn_22_h19', 'attn_23_h2', 'attn_20_h13', 'attn_23_h0', 'attn_20_h17', 'attn_23_h9', 'attn_23_h3', 'attn_20_h18', 'attn_22_h17', 'attn_21_h13', 'attn_22_h12', 'attn_20_h11', 'attn_35_h2', 'attn_22_h13'],
        [-1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0],
    )
    look_at_negatives = True

    iic_vectors = inspect_iic(
        model, 
        tokenizer, 
        steer_vec, 
        layer, 
        offset=0, 
        view_heads=True,
        view_head_dict=view_head_dict_pos,
        look_at_negatives=look_at_negatives
    )


    gemma2b_dim_heads_to_plot = {
        "15.2": [0,7,16],
        "16.1": [0, 1, 2],
        "17.-6": [1,2,5],
        "21.3": [0,1,3],
        "25.6": [0,2,4],
        "Sum": [0,1,2],
        "sv": [10,12,14]
    }

    gemma2b_ntp_heads_to_plot = {
        "15.2": [0,1,2],
        "16.1": [0, 3, 9],
        "17.7": [3, 10, 18],
        "22.1": [0, 3, 18],
        "23.0": [0, 4, 5],
        "Sum": [0,2, 18],
        "sv": [0,1,2]
    }

    gemma2b_po_heads_to_plot = {
        "15.2": [2,6,9],
        "16.1": [0, 1, 2],
        "17.-6": [0, 2, 3],
        "21.3": [0,1,2],
        "25.6": [0, 1, 2],
        "Sum": [0,1, 4],
        "sv": [0,1,3]
    }

    l3_dim_heads_to_plot = {
        "12.9": [6,7,4],
        "12.22": [1,2,17],
        "13.18": [0, 9, 11],
        "14.3": [1,5,14],
        "14.19": [1,11,12],
        "15.15": [0,1,5],
        "Sum": [0,4,17],
        "sv": [1,2,7]
    }

    l3_ntp_heads_to_plot = {
        "12.9": [1,2,6],
        "12.22": [3,10,18],
        "13.-10": [0, 1, 3],
        "13.18": [0, 3, 4],
        "14.3": [0,1,4],
        "15.15": [2,5,19],
        "Sum": [0,2,3],
        "sv": [2,9,16]
    }

    l3_po_heads_to_plot = {
        "13.18": [1, 2, 4],
        "14.3": [0,4,5],
        "15.15": [0,1,13],
        "15.-16": [0,1,2],
        "19.6": [0,1,2],
        "23.17": [0,1,2],
        "Sum": [0,4,5],
        "sv": [0,2,3]
    }

    q8_dim_heads_to_plot = {
        "20.-9": [0, 2, 6],
        "20.10": [0,2,13],
        "22.4": [0,2,3],
        "22.-6": [0,2,3],
        "23.3": [0,2,5],
        "35.2": [0,4,7],
        "Sum": [0,1,3],
        "sv": [0,3,4]
    }
    
    model_name = {
        'google/gemma-2-2b-it': 'g2',
        'meta-llama/Llama-3.2-3B-Instruct': 'l3',
        'Qwen/Qwen3-8B': 'q8',
    }[model_path]

    save_flag = True

    if model_name == 'g2' and args.learn_type == 'dim':
        heads_to_plot = gemma2b_dim_heads_to_plot
    if model_name == 'g2' and args.learn_type == 'ntp':
        heads_to_plot = gemma2b_ntp_heads_to_plot
    if model_name == 'g2' and args.learn_type == 'reps':
        heads_to_plot = gemma2b_po_heads_to_plot
    if model_name == 'l3' and args.learn_type == 'dim':
        heads_to_plot = l3_dim_heads_to_plot
    if model_name == 'l3' and args.learn_type == 'ntp':
        heads_to_plot = l3_ntp_heads_to_plot
    if model_name == 'l3' and args.learn_type == 'reps':
        heads_to_plot = l3_po_heads_to_plot
    if model_name == 'q8' and args.learn_type == 'dim':
        heads_to_plot = q8_dim_heads_to_plot

    plot(iic_vectors, steer_vec, heads_to_plot, 
         save_flag, save_path=f'svv_heatmap_{model_name}_{args.learn_type}_logit.png')
    # plot(iic_vectors, steer_vec, heads_to_plot, 
    #      save_flag, save_path=f'svv_heatmap_{model_name}_{args.learn_type}_sim.png', m='sim')

