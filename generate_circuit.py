import torch as t
from nnsight import LanguageModel
from utils.data_utils import load_steering_vector, load_refusal_completions
from data_preprocess import format_input
from transformers import AutoTokenizer, AutoModelForCausalLM
from patching.loading_utils import load_submodules, Submodule
from functools import partial
import matplotlib.pyplot as plt
from patching.attribution import get_steer_submodules
import json
from tqdm import tqdm
import os
import time
from faithfulness import forward_with_circuit, forward_base, get_input_masks

def _steer(resid_submod, steer_vec):
    resid_base = resid_submod.get_out_activation().clone()
    resid_steer = resid_base + steer_vec.to(resid_submod.device)
    resid_submod.set_out_activation(resid_steer)

def generate_steer(model, prompt_ids, steer_fn, max_new_tokens):
    # attn_weights_steer = {}
    all_logits_steer = []
    with t.no_grad(), model.generate(prompt_ids, max_new_tokens=max_new_tokens) as tracer:
        tokens = list().save()
        with tracer.iter[:]:
            steer_fn()
            logits = model.lm_head.output.save()
            token = logits[0,-1].argmax(dim=-1)
            tokens.append(token)
            all_logits_steer.append(logits[0,-1])

    return t.tensor(tokens), t.stack(all_logits_steer, dim=0)

def generate_circuit(model, prompt_ids, steer_submodules, max_new_tokens, steer_vec, edges):
    eos_token_id = model.config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_ids = {eos_token_id}
    elif isinstance(eos_token_id, list):
        eos_ids = set(eos_token_id)
    else:
        eos_ids = set()
    print(f'eos tokens: {eos_ids}')

    circuit_tokens = t.zeros(max_new_tokens, dtype=t.int)
    input_masks, node_names = get_input_masks(edges, steer_submodules, resid_submod)

    for i in range(max_new_tokens):
        base_acts, logits_base, ln_ins_base = forward_base(model, steer_submodules, prompt_ids)
        logits_circuit = forward_with_circuit(model, steer_submodules, resid_submod, steer_vec, prompt_ids, input_masks, node_names, base_acts)

        circuit_y = t.argmax(logits_circuit[0, -1], dim=-1).flatten()
        
        circuit_tokens[i] = circuit_y
        prompt_ids = t.cat((prompt_ids, circuit_y.unsqueeze(0).cpu()), dim=-1)

        if circuit_y in eos_ids:
            break

    return circuit_tokens

def dii_pass(model, prompt_ids, dii_submodules, acts_steer):
    with t.no_grad(), model.trace(prompt_ids):
        for submodule in dii_submodules:
            steer_act = acts_steer[submodule]
            submodule.set_out_activation(steer_act.to(submodule.device))
        logits_dii = model.lm_head.output.save()
    
    return logits_dii

def forward_steer(model, dii_submodules, resid_submod, prompt_ids, steer_vec):
    acts_steer = {}
    with t.no_grad(), model.trace(prompt_ids):
        resid_base = resid_submod.get_out_activation().clone()
        resid_steer = resid_base + steer_vec.to(resid_submod.device)
        resid_submod.set_out_activation(resid_steer)
        for submodule in dii_submodules:
            acts_steer[submodule] = submodule.get_out_activation().save()
        logits_steer = model.lm_head.output.save()
    return logits_steer, acts_steer

def generate_dii(model, prompt_ids, steer_submodules: list[Submodule], resid_submod, steer_vec, max_new_tokens):
    dii_tokens = t.zeros(max_new_tokens, dtype=t.int)

    dii_submodules = []
    for s in steer_submodules:
        if not s.is_attn: continue
        layer_idx = s.layer_idx
        if layer_idx >= 25: continue
        attn_v = Submodule(f'attn_{layer_idx}_v', s.submodule.v_proj, layer_idx)
        dii_submodules.append(attn_v)

    for i in range(max_new_tokens):
        logits_steer, acts_steer = forward_steer(model, dii_submodules, resid_submod, prompt_ids, steer_vec)
        logits_dii = dii_pass(model, prompt_ids, dii_submodules, acts_steer)

        dii_y = t.argmax(logits_dii[0, -1], dim=-1).flatten()
        
        dii_tokens[i] = dii_y
        prompt_ids = t.cat((prompt_ids, dii_y.unsqueeze(0).cpu()), dim=-1)


    return dii_tokens

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--harm_flag', action='store_true', default=False)
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--n', type=int, default=50)
    parser.add_argument('--method', type=str, default="simple")
    args = parser.parse_args()

    # load data
    model_path = "google/gemma-2-2b-it"
    model = LanguageModel(model_path, dispatch=True, dtype=t.float32, device_map='auto')
    model.set_attn_implementation("eager")
    print(f'model training: {model.training}')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    num_heads = model.config.num_attention_heads

    harm_flag = args.harm_flag
    layer, pos = 15, -1
    vector, coeff, steer_layer_idx = load_steering_vector("refusal", model_path, {'harm_flag': harm_flag}, pos, layer)
    print(f'loaded vector {vector.shape} with coeff {coeff} at layer {steer_layer_idx}')
    # steer_vec = t.ones((model.config.hidden_size,), dtype=t.float32, device=model.device)
    steer_vec = vector.float() * coeff

    submodules = load_submodules(model, True)
    steer_submodules, resid_submod = get_steer_submodules(submodules, layer)


    # load edges
    circuit_dir = f"circuits/{model_path.split('/')[-1]}/{args.exp_name}"
    files = [f for f in os.listdir(circuit_dir) if f.startswith(f"edges_asked{args.n}_{args.method}") and f.endswith(".json")]
    assert len(files) == 1, f"{files}"
    edgefile = files[0]
    with open(os.path.join(circuit_dir, edgefile), 'r') as edgesfile:
        print(f'Opening: {edgefile}')
        edges = json.load(edgesfile)
    
    steer_fn = partial(_steer, resid_submod=resid_submod, steer_vec=steer_vec)
    max_new_tokens= 200

    steer_flag = True
    completions = load_refusal_completions(model_path, steer_flag, {'harm_flag': harm_flag}, pos, layer)
    n_examples = min(20, len(completions))

    save_path = f"circuit_generations_{model_path.split('/')[-1]}_{'harmful' if harm_flag else 'harmless'}.jsonl"
    data = []
    start_idx = args.start_idx
    for ex in tqdm(range(start_idx, n_examples)):
        contents = completions[ex]

        prompt_inputs, prompt_strings = format_input(tokenizer, [[contents[0]]], True)
        prompt_ids = prompt_inputs['input_ids']
        bsz, seq_len = prompt_ids.shape
        print(prompt_strings)

        tik = time.time()
        steer_tokens = generate_steer(model, prompt_ids, steer_fn, max_new_tokens)
        tok = time.time()
        print(f'time to steer: {tok - tik}')
        circuit_tokens = generate_circuit(model, prompt_ids, steer_submodules, max_new_tokens, steer_vec, edges)
        # circuit_tokens = generate_dii(model, prompt_ids, steer_submodules, resid_submod, steer_vec, max_new_tokens)
        print(f'time to circuit: {time.time() - tok}')
        print(tokenizer.decode(steer_tokens[0]))
        print(f"Circuit Generation")
        print(tokenizer.decode(circuit_tokens))

        # data.append({
        #     'prompt': prompt_strings,
        #     'steer': tokenizer.decode(steer_tokens),
        #     'freeze': tokenizer.decode(freeze_tokens)
        # })
        entry = {
            'index': ex,
            'prompt': prompt_strings,
            'steer': tokenizer.decode(steer_tokens[0]),
            'circuit': tokenizer.decode(circuit_tokens)
        }
    
        with open(save_path, 'a') as f:
            f.write(json.dumps(entry, indent=4) + '\n')
        

