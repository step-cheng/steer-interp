import torch as t
from patching.loading_utils import Submodule
from nnsight import LanguageModel
from utils.data_utils import load_steering_vector, load_refusal_completions
from data_preprocess import format_input
from transformers import AutoTokenizer, AutoModelForCausalLM
from patching.loading_utils import load_submodules, Submodule
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import seaborn as sns

if __name__ == '__main__':

    model_path = "google/gemma-2-2b-it"
    model = LanguageModel(model_path, dispatch=True, dtype=t.float32, device_map='auto')
    model.set_attn_implementation("eager")
    print(f'model training: {model.training}')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    num_heads = model.config.num_attention_heads

    harm_flag = False
    # learned_vector_info = t.load('learned_refusal_sv_lrdecrease.pt', map_location='cpu')
    # vector_learn = learned_vector_info['vector'].to(model.device)

    layer, pos = 15, -1
    # vector_l10, coeff, steer_layer_idx = load_steering_vector("refusal", model_path, {'harm_flag': harm_flag}, -5, 10)
    # print(f'loaded vector {vector_l10.shape} with coeff {coeff} at layer {steer_layer_idx}, norm {vector_l10.norm()}')
    # vector_l14, _, _ = load_steering_vector("refusal", model_path, {'harm_flag': harm_flag}, -2, 14)
    # print(f'loaded vector {vector_l14.shape} with coeff {coeff} at layer 14, norm {vector_l14.norm()}')
    vector, coeff, steer_layer_idx = load_steering_vector("refusal", model_path, {'harm_flag': harm_flag}, -1, 15)
    print(f'loaded vector {vector.shape} with coeff {coeff} at layer 15, norm {vector.norm()}')
    # layer, pos = 23, -1
    # vector, coeff, steer_layer_idx = load_steering_vector("refusal", model_path, {'harm_flag': harm_flag}, pos, layer)
    # print(f'loaded vector {vector.shape} with coeff {coeff} at layer 15, norm {vector.norm()}')

    # print(f'Similarity 15 14: {t.cosine_similarity(vector_l15.cpu(), vector_l14.cpu(), dim=0)}')
    # print(f'Similarity 15 10: {t.cosine_similarity(vector_l15.cpu(), vector_l10.cpu(), dim=0)}')
    # print(f'Similarity 10 14: {t.cosine_similarity(vector_l10.cpu(), vector_l14.cpu(), dim=0)}')
    steer_vec = vector.float() * coeff
    # steer_vec = vector_learn.float() * coeff
    submodules = load_submodules(model, True)
    resid_submod = submodules.resids[steer_layer_idx-1]
    attn_steer_submodules = [] # because steering is applied as a posthook...
    for j in range(steer_layer_idx, len(submodules.attns)):
        attn_steer_submodules.append(submodules.attns[j])
        
    steer_flag = True
    completions = load_refusal_completions(model_path, steer_flag, {'harm_flag': harm_flag}, pos, layer)
    n_examples = min(100, len(completions))
    n_examples = 0

    class LN_Proxy:
        def __init__(self, ln):
            self.eps = ln.eps
            self.weight = ln.weight

        def get_norm_stats(self, x):
            return t.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        
        def get_norm_v(self, raw_in_v, ln_stats):
            in_v = raw_in_v.float() * ln_stats
            in_v = in_v * (1.0 + self.weight.float())
            return in_v.type_as(raw_in_v)
        
    # check attention freeze
    special_token_difference = 0
    real_token_difference = 0
    total_difference = 0
    for ex in range(n_examples):
        contents = completions[ex]

        prompt_inputs, prompt_strings = format_input(tokenizer, [[contents[0]]], True)
        prompt_ids = prompt_inputs['input_ids']
        
        # prompt_ids = t.cat((prompt_ids, t.tensor([[235285]])), dim=-1)
        bsz, seq_len = prompt_ids.shape
        toks_decoded = [tokenizer.decode(id) for id in prompt_ids[0]]

        attn_weights_base = {}
        acts_base = {}
        ln_stats = {}
        acts_post_attn_ln = {}
        with t.no_grad(), model.trace(prompt_ids):
            for attn in attn_steer_submodules:
                layer_idx = int(attn.name.split('_')[1])
                resid = model.model.layers[layer_idx]
                
                attn_out_tuple = attn.submodule.source.attention_interface_0.output
                attn_weights_base[attn] = attn_out_tuple[1].save()
                head_outs = attn.get_per_head_outputs()
                acts_base[attn] = head_outs.save()

                post_attn_ln = resid.post_attention_layernorm
                ln_proxy = LN_Proxy(post_attn_ln)
                inp = post_attn_ln.input.save()
                stats = ln_proxy.get_norm_stats(inp)
                ln_stats[attn] = (stats, ln_proxy)
                acts_post_attn_ln[attn] = post_attn_ln.output.save()

            logits_base = model.lm_head.output.save()


        for attn in attn_steer_submodules:
            acts_pre_post_ln = acts_base[attn]
            stats, ln_proxy = ln_stats[attn]
            print(stats.shape, acts_pre_post_ln.shape)
            acts_norm = ln_proxy.get_norm_v(acts_pre_post_ln, stats.unsqueeze(dim=-1)).sum(dim=2)
            print(acts_norm)
            print(acts_post_attn_ln[attn])
            assert t.allclose(acts_norm, acts_post_attn_ln[attn], atol=1e-5), attn.name


        exit()

        attn_weights_steer = {}
        with t.no_grad(), model.trace(prompt_ids):
            resid_base = resid_submod.get_out_activation().clone()
            resid_steer = resid_base + steer_vec.to(resid_submod.device)
            resid_submod.set_out_activation(resid_steer)
            for attn in attn_steer_submodules:
                attn_out_tuple = attn.submodule.source.attention_interface_0.output
                attn_weights_steer[attn] = attn_out_tuple[1].save()
            logits_steer = model.lm_head.output.save()

        # freeze attn weights
        with t.no_grad(), model.trace(prompt_ids):
            resid_base = resid_submod.get_out_activation().clone()
            resid_steer = resid_base + steer_vec.to(resid_submod.device)
            resid_submod.set_out_activation(resid_steer)
            for attn in attn_steer_submodules[1:]:
                attn_base = attn_weights_base[attn]

                values = attn.submodule.v_proj.output
                b, n = prompt_ids.shape
                values = values.view(b, n, -1, attn.submodule.head_dim).transpose(1,2)
                values = t.repeat_interleave(values, dim=1, repeats=attn.submodule.num_key_value_groups)
                new_attn_out = t.matmul(attn_base, values)
                new_attn_out = new_attn_out.transpose(1,2).contiguous()
                new_attn_out = new_attn_out.reshape(b,n,-1).contiguous()
                attn.submodule.o_proj.input = new_attn_out

            logits_freeze = model.lm_head.output.save()
        
        base_ys = t.argmax(logits_base[:,-1], dim=-1)
        steer_ys = t.argmax(logits_steer[:,-1], dim=-1)
        froze_ys = t.argmax(logits_freeze[:,-1], dim=-1)

        # print(f"base ys: {tokenizer.decode(base_ys)}, token {base_ys.item()}")
        # print(f"steer ys: {tokenizer.decode(steer_ys)}, token {steer_ys.item()}")
        # print(f"froze ys: {tokenizer.decode(froze_ys)}, token {froze_ys.item()}")

        attn_deltas = {}
        attn_steer = {}
        attn_base = {}
        for i, attn in enumerate(attn_steer_submodules):
            attn_delta = attn_weights_steer[attn] - attn_weights_base[attn]
            for h in range(num_heads):
                attn_deltas[f"{attn.name}_h{h}"] = attn_delta[0, h]
                attn_steer[f"{attn.name}_h{h}"] = attn_weights_steer[attn][0, h]
                attn_base[f"{attn.name}_h{h}"] = attn_weights_base[attn][0, h]


        # <bos> <start_of_turn> user \n | <end-of-turn> \n start of turn model \n
        for head, deltas in attn_deltas.items():
            last_tok_delta = deltas[-1]
            total_difference += t.sum(last_tok_delta.abs()).item()
            special_token_difference += t.sum(last_tok_delta[:4].abs()).item() + t.sum(last_tok_delta[-5:].abs()).item()
            real_token_difference += t.sum(last_tok_delta[4:-5].abs()).item()


        # plot base attention values
        # attn_head_names = list(attn_base.keys())
        # attn_base_img = np.zeros((len(attn_head_names),seq_len))
        # for i, head in enumerate(attn_head_names):
        #     attn_base_img[i] = attn_base[head][-1].cpu().numpy()
        
        # plt.figure(figsize=(10,30))
        # plt.imshow(attn_base_img, aspect="auto")
        # plt.xticks(range(seq_len), toks_decoded, rotation=90)
        # plt.yticks(range(len(attn_head_names)), attn_head_names)
        # plt.colorbar()
        # plt.tight_layout()
        # plt.savefig(f"images/{ex}_attn_base_{'harmful' if harm_flag else 'harmless'}.png")
        # plt.close()
        
        # # plot model token
        # top_attn_head_names = sorted(attn_head_names, key=lambda k: t.norm(attn_deltas[k][-2]), reverse=True)[:len(attn_head_names)//4]
        # top_attn_head_names = sorted(top_attn_head_names)
        # attn_deltas_img = np.zeros((len(attn_head_names),seq_len))
        # for i, head in enumerate(attn_head_names):
        #     attn_deltas_img[i] = attn_steer[head][-2].cpu().numpy()
        
        # plt.figure(figsize=(10,30))
        # plt.imshow(attn_deltas_img, aspect="auto")
        # plt.xticks(range(seq_len), toks_decoded, rotation=90)
        # plt.yticks(range(len(attn_head_names)), attn_head_names)
        # plt.colorbar()
        # plt.tight_layout()
        # plt.savefig(f"images/{ex}_attn_pos-2_steer_{'harmful' if harm_flag else 'harmless'}.png")
        # plt.close()

        # plot deltas
        # attn_head_names = list(attn_base.keys())
        # top_attn_head_names = sorted(attn_head_names, key=lambda k: t.norm(attn_deltas[k][-1]), reverse=True)[:len(attn_head_names)//4]
        # top_attn_head_names = sorted(top_attn_head_names)
        # attn_deltas_img = np.zeros((len(top_attn_head_names),seq_len))
        # for i, head in enumerate(top_attn_head_names):
        #     attn_deltas_img[i] = attn_deltas[head][-1].cpu().numpy()

        # attn_head_names_display = [
        #     f"L{name.split('_')[1]}H{name.split('_')[2][1]}" for name in top_attn_head_names
        # ]
        # toks_decoded_display = [tok.replace('\n', '\\n') for tok in toks_decoded]
        
        # plt.figure(figsize=(15,12))
        # sns.heatmap(
        #     attn_deltas_img,
        #     xticklabels=toks_decoded_display,
        #     yticklabels=attn_head_names_display,
        #     cmap='RdBu_r',  # Diverging colormap centered at 0
        #     center=0,
        #     linewidths=0,
        #     square=False,
        #     cbar=False,
        # )
        # plt.xticks(rotation=45, fontsize=14)
        # plt.yticks(rotation=0)
        # # plt.colorbar()
        # plt.tight_layout()
        # plt.savefig(f"images/{ex}_attn_deltas_{'harmful' if harm_flag else 'harmless'}_topquarter.png")
        # plt.close()
    
    # ratio1 = special_token_difference / total_difference
    # ratio2 = real_token_difference / total_difference
    # print(f'ratio: {ratio1}, ratio: {ratio2}')