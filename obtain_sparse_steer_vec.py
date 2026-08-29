import torch as t
import json
import os
from config_act_patch import Config
import argparse
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils.data_utils import load_steering_vector
from config_act_patch import dim_layer_map, dim_layer_pos_dict

t.manual_seed(42)

def sanity_check_values(nodes, edges):
    for k, v in nodes.items():
        assert t.isnan(v).sum() == 0, f"nan at {k}"
    for up in edges:
        for down in edges[up]:
            assert t.isnan(edges[up][down]).sum() == 0, f"nan at edge {up} to {down}"
    print('passed nan sanity check')

def inspect_resid_edges(edges, nodes):
    for name in nodes:
        if name.startswith('resid'):
            layer = int(name.split('_')[1])+1
            break
    effect = nodes[f'resid_{layer-1}'].float()

    resid_edges = []
    for edge in edges:
        if edge[0] == f'resid_{layer-1}':
            resid_edges.append(edge)
    
    print(len(resid_edges))

def compare_similarity(v1, v2):
    v1_u = v1 / v1.norm()
    v2_u = v2 / v2.norm()
    return t.dot(v1_u, v2_u)

def get_resid_effect(nodes):
    for name in nodes:
        if name.startswith('resid'):
            layer = int(name.split('_')[1])+1
            break
    effect = nodes[f'resid_{layer-1}'].float()
    return effect

def save_prop_histogram(effect, steer_vec):
    effect_norm = effect.norm()
    vector = steer_vec.cpu().float()
    vector_norm = vector.norm()

    effect_unit = effect / effect_norm
    vector_unit = vector / vector_norm
    props = np.abs(effect_unit) / np.abs(vector_unit)
    hist, bin_edges = np.histogram(props, 10)
    print(hist, bin_edges)
    fig, axs = plt.subplots()
    axs.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), align='edge', edgecolor='black')
    axs.set_xlabel("effect / vector for unit normalized effect and vector")
    axs.set_ylabel("count")
    fig.savefig('prop_histogram.png')

def get_naive_rtol_vector(effect, steer_vec, rtol, save=False, save_prefix=None):
    pass

def get_mag_vector(effect, steer_vec, n, save=False, save_prefix=None):
    vector_norm = steer_vec.norm()
    
    weak_pos = 0
    weak_neg = 0
    sorted_dims = t.argsort(steer_vec.abs())
    weak_dims = sorted_dims[:n]
    mag_vector = steer_vec.clone()
    for i in range(n):
        if mag_vector[sorted_dims[i]] > 0:
            weak_pos += 1
        else:
            weak_neg += 1
        mag_vector[sorted_dims[i]] = 0

    print(f'num weak dims: {len(weak_dims)}, num weak pos: {weak_pos}, num weak neg: {weak_neg}')

    print(f'similarity drop bottom {n}: {compare_similarity(steer_vec, mag_vector)}')

    if save:
        save_path = f"{save_prefix}_mag={n}.pt"
        base_path = os.path.dirname(save_path)
        os.makedirs(base_path, exist_ok=True)
        data = {
            'layer': layer,
            'direction': mag_vector * (vector_norm/mag_vector.norm())
        }
        t.save(data, save_path)
    return mag_vector

def get_rtol_vector(effect, steer_vec, rtol, save=False, save_prefix=None, renorm=True):
    effect_norm = effect.norm()
    vector_norm = steer_vec.norm()
    effect_unit = effect / effect_norm
    vector_unit = steer_vec / vector_norm
    
    weak_pos = 0
    weak_neg = 0
    weak_dims = []
    for i, (e, v) in enumerate(zip(effect_unit, vector_unit)):
        if abs(e) < abs(v*rtol): 
            weak_dims.append(i)
            if e > 0: weak_pos += 1
            if e < 0: weak_neg += 1

    # check the values of the dims of s that are kept
    kept_dims = set(list(range(len(steer_vec)))) - set(weak_dims)
    print(f'number of kept dims: {len(kept_dims)}, num total: {len(steer_vec)}, num weak: {len(weak_dims)}')
    mean_abs_dim = steer_vec.abs().mean()
    std_abs_dim = steer_vec.abs().std()
    print(f'mean std of all steer_vec dims: {mean_abs_dim}, {std_abs_dim}')
    print(f'mean std of kept steer_vec dims: {steer_vec[list(kept_dims)].abs().mean()}, {steer_vec[list(kept_dims)].abs().std()}')
    print(f'num kept dims below one std of dense steer_vec: {(steer_vec[list(kept_dims)].abs() < mean_abs_dim - std_abs_dim).sum()}')
    print(f'num kept dims below two stds of dense steer_vec: {(steer_vec[list(kept_dims)].abs() < mean_abs_dim - 2*std_abs_dim).sum()}')


    print(f'rtol: {rtol}, num weak dims: {len(weak_dims)}, num weak pos: {weak_pos}, num weak neg: {weak_neg}')
    rtol_vector = steer_vec.clone()
    for dim in weak_dims:
        rtol_vector[dim] = 0

    print(f'similarity rtol={rtol}: {compare_similarity(steer_vec, rtol_vector)}')

    if save:
        save_path = f"{save_prefix}_rtol={rtol}.pt"
        base_path = os.path.dirname(save_path)
        os.makedirs(base_path, exist_ok=True)
        data = {
            'layer': layer,
            'direction': rtol_vector * (vector_norm/rtol_vector.norm()) if renorm else rtol_vector
        }
        t.save(data, f"{save_prefix}_rtol={rtol}{('_norenorm' if not renorm else '')}.pt")
    return rtol_vector, len(weak_dims)

def get_ie_vector(effect, steer_vec, n, save=False, save_prefix=None):
    print(f'getting ie vector {n}')
    vector_norm = steer_vec.norm()
    
    weak_pos = 0
    weak_neg = 0
    sorted_dims = t.argsort(effect.abs())
    weak_dims = sorted_dims[:n]
    ie_vector = steer_vec.clone()
    for i in range(n):
        if ie_vector[sorted_dims[i]] > 0:
            weak_pos += 1
        else:
            weak_neg += 1
        ie_vector[sorted_dims[i]] = 0

    print(f'num weak dims: {len(weak_dims)}, num weak pos: {weak_pos}, num weak neg: {weak_neg}')

    print(f'similarity drop bottom {n}: {compare_similarity(steer_vec, ie_vector)}')

    if save:
        save_path = f"{save_prefix}_ie={n}.pt"
        base_path = os.path.dirname(save_path)
        os.makedirs(base_path, exist_ok=True)
        data = {
            'layer': layer,
            'direction': ie_vector * (vector_norm/ie_vector.norm())
        }
        t.save(data, save_path)
    return ie_vector
    
def get_sparse_vector(effect, steer_vec, p, save=False, save_prefix=None):
    effect_norm = effect.norm()
    vector_norm = steer_vec.norm()
    effect_unit = effect / effect_norm
    vector_unit = vector / vector_norm
    weak_pos = 0
    weak_neg = 0
    props = effect_unit / vector_unit

    sorted_inds = t.argsort(props)
    num_remove = t.floor(len(sorted_inds) * p)

    sparse_vector = vector.clone()
    for ind in sorted_inds[:num_remove]:
        if sparse_vector[ind] > 0:
            weak_pos += 1
        elif sparse_vector[ind] < 0:
            weak_neg += 1
        sparse_vector[ind] = 0

    print(f'similarity sparse={p}: {compare_similarity(vector, sparse_vector)}')

    if save:
        data = {
            'layer': layer,
            'direction': sparse_vector * (vector_norm/sparse_vector.norm())
        }
        t.save(data, f"vectors/{save_prefix}_sparse={p}.pt")
    return sparse_vector

def get_sign_flip_vector(effect, steer_vec, p, save=False, save_prefix=None):
    # flip sign for any dim with negative effect
    vector_norm = steer_vec.norm()
    effect_signs = np.sign(effect)
    sign_vector = effect_signs * vector
    print(f'similarity sign flip: {compare_similarity(vector, sign_vector)}')

    if save:
        data = {
            'layer': 15,
            'direction': sign_vector * (vector_norm/sign_vector.norm())
        }
        t.save(data, f"vectors/{save_prefix}_sign.pt")

def get_noneg_vector(effect, steer_vec, p, save=False, save_prefix=None):
    # remove any dim with negative effect
    vector_norm = steer_vec.norm()
    print(f'num neg dims {len(t.argwhere(effect < 0).flatten())}')
    pos_vector = vector.clone()
    for i in range(len(pos_vector)):
        if effect[i] < 0:
            pos_vector[i] = 0
    print(f'similarity noneg: {compare_similarity(vector, pos_vector)}')

    if save:
        data = {
            'layer': 15,
            'direction': pos_vector * (vector_norm/pos_vector.norm())
        }
        t.save(data, f"vectors/{save_prefix}_noneg.pt")

def get_random_dropout_vector(steer_vec, n, save=False, save_prefix=None):
    vector_norm = steer_vec.norm()
    dropout_vector = steer_vec.clone()
    drop_inds = t.randperm(len(dropout_vector))[:n]
    dropout_vector[drop_inds] = 0

    print(f'dropout n={n}: {compare_similarity(steer_vec, dropout_vector)}')

    if save:
        save_path = f"{save_prefix}_dropout={n}.pt"
        base_path = os.path.dirname(save_path)
        os.makedirs(base_path, exist_ok=True)
        data = {
            'layer': layer,
            'direction': dropout_vector * (vector_norm / dropout_vector.norm())
        }
        t.save(data, save_path)
    return dropout_vector

def get_edge_list(edges: dict):
    edges_listed = []
    up_names = set()
    down_names = set()
    for up in edges:
        up_names.add(up)
        for down in edges[up]:
            down_names.add(down)
            edges_listed.append((up, down, edges[up][down].sum().item()))

    return edges_listed, up_names, down_names

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--model_path', type=str, default='google/gemma-2-2b-it')
    parser.add_argument('--learn_type', type=str, choices=['dim', 'ntp', 'reps', 'ortho'])
    parser.add_argument('--learn_path', type=str, default=None)
    parser.add_argument('--save', action="store_true", default=False)
    args = parser.parse_args()
    exp_name = args.exp_name

    model_path = args.model_path

    n_layers = {
        "google/gemma-2b-it": 18,
        "google/gemma-2-2b-it": 26,
        "google/gemma-2-9b-it": 42,
        "meta-llama/Llama-3.1-8B-Instruct": 32,
        "meta-llama/Llama-3.2-3B-Instruct": 28,
        "Qwen/Qwen3-8B": 36
    }[model_path]
    layer = dim_layer_map[model_path]
    _, pos = dim_layer_pos_dict[model_path][layer]
    

    circuit_dir1 = f"circuits/{args.model_path.split('/')[-1]}/{exp_name}"
    circuit_dir2 = f"/fs/nexus-scratch/scheng03/steer-interp-results/{circuit_dir1}"
    if os.path.exists(f"{circuit_dir1}"):
        circuit_dir = circuit_dir1
    elif os.path.exists(f"{circuit_dir2}"):
        circuit_dir = circuit_dir2
    else:
        print(f"save dirs {circuit_dir1}, {circuit_dir2} do not exist")
    try:
        data = t.load(os.path.join(circuit_dir, "patching_results.pt"), map_location=t.device("cpu"))
    except Exception as e:
        print('could not find data, assuming this is an aggregated exp')
        print(f'{e}')
        exit()

    nodes = data['nodes']
    edges = data['edges']
    total_poi = data['num_positions']

    sanity_check_values(nodes, edges)
    steer_vec, coeff, layer = load_steering_vector(
        "refusal", 
        args.model_path, 
        {'harm_flag': False}, 
        pos, 
        layer,
        (True if args.learn_type != 'dim' else False),
        vector_base_path=args.learn_path
        )
    
    save_dir = f"vectors/{model_path.split('/')[-1]}"
    save_prefix = f"{save_dir}/{args.exp_name}"
    os.makedirs(save_dir, exist_ok=True)

    effect = get_resid_effect(nodes).cpu().float()
    steer_vec = steer_vec.cpu().float()

    for r in [0.1, 0.3, 0.5, 1, 1.5, 2, 2.5]:
        rtol_vector, n_drop = get_rtol_vector(effect, steer_vec, r, save=True, save_prefix=save_prefix, renorm=True)
        print(f'with rtol {r}, dropped {n_drop} dimensions')
        # ie_vector = get_ie_vector(effect, steer_vec, n_drop, save=True, save_prefix=save_prefix)
        # dropout_vector = get_random_dropout_vector(steer_vec, n_drop, True, save_prefix=save_prefix)
        # mag_vector = get_mag_vector(effect, steer_vec, n_drop, True, save_prefix=save_prefix)
    

    # edges_list, up_names, down_names = get_edge_list(edges)
    # inspect_resid_edges(edges_list, nodes)
    # inspect_circuit(pruned_circuit_edges, up_names, down_names, edges_list)
