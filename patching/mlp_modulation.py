from nnsight import LanguageModel
from .loading_utils import Submodule, load_submodules, SubmoduleStash
from .attribution import get_steer_submodules
from utils.data_utils import load_steering_vector, load_refusal_completions
from data_preprocess import format_input
import torch as t
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

def run_steer():
    pass

def run_base():
    pass

def logit_lens(model, vector):
    final_ln = model.model.norm
    lm_head = model.lm_head

    return lm_head(final_ln(vector.to(final_ln.device)))

def retrieve_value_vector(model, layer, pos, offset):
    mlp = model.model.layers[layer+offset].mlp
    down_proj = mlp.down_proj
    vector = down_proj.weight[:,pos]
    
    return vector.unsqueeze(0)

def get_mlp_modules(submodules: SubmoduleStash, layer_idx):
    ret = []
    for i in range(len(submodules.mlps)):
        if i >= layer_idx:
            ret.append(submodules.mlps[i])
    return ret

def analyze_act_diffs(act_diffs, norm_path='mlp_norm', histogram_path='mlp_histogram'):
    diff_norms = act_diffs.norm(dim=-1)
    fig, axs = plt.subplots()
    # axs.bar(range(layer_idx, model.config.num_hidden_layers), diff_norms.cpu().flatten())
    # fig.savefig(f'{norm_path}_{harm_flag}.png')

    all_indices = [[i, p] for i in range(act_diffs.shape[0]) for p in range(act_diffs.shape[1])]
    sorted_indices = sorted(all_indices, key=lambda ind: act_diffs[*ind].abs(), reverse=True)
    topk_acts = sorted_indices[:10]
    print(f'topk acts: {topk_acts}')

    for top_act in topk_acts:
        value_vector = retrieve_value_vector(model, top_act[0], top_act[1], layer_idx)
        logits = logit_lens(model, value_vector)

        vals, inds = t.topk(logits, k=10, dim=-1)
        print(f"layer {top_act[0]+layer_idx}, act {top_act[1]}, value {act_diffs[*top_act]}: {'|'.join([tokenizer.decode(ind) for ind in inds])}")
    
    # make histogram
    hist, bin_edges = t.histogram(act_diffs.flatten(), 10)
    print(hist)
    print(bin_edges)
    widths = t.diff(bin_edges)
    x_coords = bin_edges[:-1]
    fig, axs = plt.subplots()
    axs.bar(x=x_coords, height=t.clamp(hist, max=20), width=widths, align="edge", edgecolor='black')
    fig.savefig(f'{histogram_path}_{harm_flag}.png')

def measure_input_invariant_modulation():
    act_base = []
    act_steer = []
    with t.no_grad(), model.trace([[40]]):
        for mlp in mlp_modules:
            hidden_acts = mlp.submodule.down_proj.input
            act_base.append(hidden_acts.squeeze(0).squeeze(0).save())
    
    # check each mlp individually
    for mlp in mlp_modules:
        with t.no_grad(), model.trace([[40]]):
            inp = mlp.pre_ln.input
            new_inp = inp + coeff*vector.to(mlp.submodule.device)
            mlp.pre_ln.input = new_inp

            hidden_acts = mlp.submodule.down_proj.input
            act_steer.append(hidden_acts.squeeze(0).squeeze(0).save())
    
    act_diffs = t.stack(act_steer) - t.stack(act_base)
    print(act_diffs.shape)
    analyze_act_diffs(act_diffs)


def measure_modulation_first_tok():
    # loading steered completions
    completions = load_refusal_completions(model_path, True, {'harm_flag': harm_flag}, pos, layer_idx)
    all_acts_base = []
    all_acts_steer = []
    for completion in tqdm(completions[:10]):
        prompt_inputs, prompt_strings = format_input(tokenizer, [[completion[0]]], True)
        prompt_ids = prompt_inputs['input_ids']
        # print([tokenizer.decode(id) for id in prompt_ids[0]])

        acts_base_sample = []
        with t.no_grad(), model.trace(prompt_ids):
            for mlp in mlp_modules:
                hidden_acts = mlp.submodule.down_proj.input
                acts_base_sample.append(hidden_acts[0,-1].save())
            
            logits_base = model.lm_head.output[0,-1].save()
        # print(f'base pred: {tokenizer.decode(t.argmax(logits_base,dim=-1))}')

        acts_steer_sample = []
        with t.no_grad(), model.trace(prompt_ids):
            resid_base_act = steer_submod.get_out_activation()
            resid_base_steer = resid_base_act + coeff*vector.to(steer_submod.device)
            steer_submod.set_out_activation(resid_base_steer)
            for mlp in mlp_modules:
                hidden_acts = mlp.submodule.down_proj.input
                acts_steer_sample.append(hidden_acts[0,-1].save())
            
            logits_steer = model.lm_head.output[0,-1].save()
        # print(f'steer pred: {tokenizer.decode(t.argmax(logits_steer,dim=-1))}')

        all_acts_base.append(t.stack(acts_base_sample))
        all_acts_steer.append(t.stack(acts_steer_sample))

    # num_examples mlp_modules dim
    all_acts_base = t.stack(all_acts_base)
    all_acts_steer = t.stack(all_acts_steer)

    act_diffs = all_acts_steer - all_acts_base
    print(act_diffs.shape)
    act_diffs = act_diffs.mean(dim=0)
    print(act_diffs.shape)

    analyze_act_diffs(act_diffs, "mlp_norm_first", "mlp_histogram_first")
    


if __name__ == '__main__':

    model_path = 'google/gemma-2-2b-it'
    dtype = t.float32
    model = LanguageModel(model_path, dispatch=True, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    layer_idx, pos = 15, -1
    harm_flag = True
    vector, coeff, _ = load_steering_vector('refusal', model_path, {'harm_flag': harm_flag}, pos, layer_idx)
    vector = vector.to(dtype)
    print(f"using coefficient {coeff}")

    submodules = load_submodules(model)
    mlp_modules = get_mlp_modules(submodules, layer_idx)
    steer_submod = submodules.resids[layer_idx-1]

    measure_input_invariant_modulation()

    measure_modulation_first_tok()

    



    
