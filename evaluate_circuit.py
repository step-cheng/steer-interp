import torch as t
from utils.plotting_utils import plot_module_circuit, plot_module_circuit_granular
import json
import os
from config_act_patch import Config
import argparse
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

t.manual_seed(42)

def sanity_check_values(nodes, edges):
    for k, v in nodes.items():
        assert t.isnan(v).sum() == 0, f"nan at {k}"
    for up in edges:
        for down in edges[up]:
            assert t.isnan(edges[up][down]).sum() == 0, f"nan at edge {up} to {down}"
    print('passed nan sanity check')

def inspect_nodes(nodes):
    vals = []
    neg_nodes = []
    total_effect = None
    for key in nodes.keys():
        if key == 'y': 
            total_effect = nodes['y']
            continue
        if key == 'lm_head':
            continue
        vals.append(nodes[key])    
        sum_val = nodes[key].sum()
        if sum_val < 0:
            if len(key.split('_')) == 3:
                h_pos = key.split('_')[2]
            else:
                h_pos = None
            neg_nodes.append((key, h_pos))
            # print(f'negative node {key} w/ value: {sum_val}')
    
    sorted_attn_keys = list(key for key in nodes if 'attn' in key)
    sorted_attn_keys = sorted(sorted_attn_keys, key=lambda x: abs(nodes[x].sum()), reverse=True)
    print(f"top 25 of the {len(sorted_attn_keys)} heads: {sorted_attn_keys[:25]}")
    print(f"{[t.sign(nodes[n].sum()).item() for n in sorted_attn_keys[:25]]}")

    print(f'total effect: {total_effect}')
    vals = t.stack(vals, dim=0)
    print(f'all nodes shape: {vals.shape}')

    elem_mean_effect = vals.mean()
    elem_std_effect = vals.std()
    elem_max_effect = vals.max()
    elem_min_effect = vals.min()
    print(f"""Element-wise: {elem_mean_effect} +/- {elem_std_effect}, min {elem_min_effect}, max {elem_max_effect}""")

    node_mean_effect = vals.sum(dim=-1).mean(dim=0)
    node_std_effect = vals.sum(dim=-1).std(dim=0)
    node_max_effect = vals.sum(dim=-1).max(dim=0)[0]
    node_min_effect = vals.sum(dim=-1).min(dim=0)[0]
    print(f"""Node-wise: {node_mean_effect} +/- {node_std_effect}, min {node_min_effect}, max {node_max_effect}""")

    mean_effect_per_dim = vals.mean(dim=0)
    std_effect_per_dim = vals.std(dim=0)
    dim_mean_effect = mean_effect_per_dim.mean()
    dim_std_effect = mean_effect_per_dim.std()
    dim_max_effect, dim_max = mean_effect_per_dim.max(dim=0)
    dim_min_effect, dim_min = mean_effect_per_dim.min(dim=0)
    print(f"""Dim-wise: {dim_mean_effect} +/- {dim_std_effect}, 
          min {dim_min_effect} +/- {std_effect_per_dim[dim_min]} at dim {dim_min}, 
          max {dim_max_effect} +/- {std_effect_per_dim[dim_max]} at dim {dim_max}""")

    return neg_nodes

def inspect_edges(edges):
    vals = []
    neg_edges = []
    for up in edges:
        for down in edges[up]:
            vals.append(edges[up][down])
            sum_val = edges[up][down].sum()
            if sum_val < 0:
                if len(up.split('_')) == 3:
                    h_pos = up.split('_')[2]
                else: 
                    h_pos = None
                if len(down.split('_')) == 4:
                    h_pos = down.split('_')[2]
                    qkv_input = down.split('_')[3]
                else:
                    qkv_input = None
                    h_pos = None
                neg_edges.append((up, h_pos, down, qkv_input))
                # print(f'negative edge {up} to {down} with val {sum_val}')

    vals = t.stack(vals, dim=0)
    print(f'all edges shape: {vals.shape}')
    elem_mean_effect = vals.mean()
    elem_std_effect = vals.std()
    elem_max_effect = vals.max()
    elem_min_effect = vals.min()
    edge_mean_effect = vals.sum(dim=-1).mean(dim=0)
    edge_std_effect = vals.sum(dim=-1).std(dim=0)
    edge_max_effect = vals.sum(dim=-1).max(dim=0)[0]
    edge_min_effect = vals.sum(dim=-1).min(dim=0)[0]

    print(f"""Element-wise: {elem_mean_effect} +/- {elem_std_effect}, min {elem_min_effect}, max {elem_max_effect}""")
    print(f"""Edge-wise: {edge_mean_effect} +/- {edge_std_effect}, min {edge_max_effect}, max {edge_min_effect}""")

    return neg_edges

def get_upstream_base(name):
    """                                                                                                                                                                                                                                                                 
    Returns the circuit vertex key for a downstream edge target.                                                                                                                                                                                                        
    ig:  'attn_10_h5' -> 'attn_10'      (whole module is the vertex)
    ig2: 'attn_10_h5' -> 'attn_10_h5'   (specific head is the vertex)                                                                                                                                                                                                 
    other: 'mlp_10'     -> 'mlp_10'                                                                                                                                                                                                                                     
    """                                                                                                                                                                                                                                                                 
    parts = name.split('_')                                                                                                                                                                                                                                             
    if parts[0] == 'attn' and GRANULAR:
        return name                # attn_l_hN                                                                                                                                                    
    elif parts[0] == 'attn' and not GRANULAR:
        return '_'.join(parts[:2])                # attn_l                                                                                                                                                                                                              
    return name 
 
def get_downstream_base(name):
    """                                                                                                                                                                                                                                                                 
    Returns the circuit vertex key for a downstream edge target.                                                                                                                                                                                                        
    ig:  'attn_10_q'    -> 'attn_10'      (whole module is the vertex)
    ig2: 'attn_10_h5_q' -> 'attn_10_h5'   (specific head is the vertex)                                                                                                                                                                                                 
    ig2: 'attn_10_h2_k' -> 'attn_10_h4', 'attn_10_h5'   (specific head is the vertex)                                                                                                                                                                                                 
    other: 'mlp_10'     -> 'mlp_10'                                                                                                                                                                                                                                     
    """                                                                                                                                                                                                                                                                 
    parts = name.split('_')                                                                                                                                                                                                                                             
    if parts[0] == 'attn' and GRANULAR:
        assert len(parts) == 4, 'check if granular or not'   # ig2: attn_l_hN_qkv                                                                                                                                                                                                   
        if parts[3] == 'q':
            return '_'.join(parts[:3])                # attn_l_hN                                                                                                                                                    
        if parts[3] in ['k', 'v']:
            # handle groups
            group_number = int(parts[2][1:])
            names = []
            for i in range(head_kv_group_ratio):
                h = head_kv_group_ratio*group_number + i
                names.append(f"{parts[0]}_{parts[1]}_h{h}")
            return names                 # attn_l_hN                                                                                                                                                                                                            
    elif parts[0] == 'attn' and not GRANULAR:
        assert len(parts) == 3 and parts[2] in ('q','k','v'), 'check if granular or not'  # ig: attn_l_qkv                                                                                                                                                                        
        return '_'.join(parts[:2])                # attn_l                                                                                                                                                                                                              
    return name  

def build_forward(edges, input_node, output_node, n):
    """
    Greedy circuit search algorithm starting from logits.
    
    Args:
        edges: dict of dict, where edges[upstream][downstream] = score
        n: number of edges to select
        logits_node: name of the logits node (default 'lm_head')
    
    Returns:
        list of tuples (upstream, downstream, score)
    """
    C_V = {input_node}  # Circuit vertices
    C_E = set()  # Circuit edges (as tuples)
    
    for i in range(n):
        # Find candidate edges: edges not in circuit whose child is in circuit
        D = []
        for upstream, downstream, score in edges:
            edge_tuple = (upstream, downstream)
            upstream_base = get_upstream_base(upstream)
            if edge_tuple not in C_E and upstream_base in C_V:
                D.append((upstream, downstream, score))
        
        if not D:
            break  # No more edges to add
        
        # Select edge with highest absolute score
        best_edge = max(D, key=lambda x: abs(x[2]))
        upstream, downstream, score = best_edge
        upstream_base = get_upstream_base(upstream)
        downstream_base = get_downstream_base(downstream)
        
        # Add edge and parent node to circuit
        C_E.add((upstream, downstream))
        if type(downstream_base) == str:
            C_V.add(downstream_base)
        elif type(downstream_base) == list:
            C_V.update(downstream_base)
        
    # Return edges as list of tuples with their scores
    return [(up, down, score) for up, down, score in edges if (up, down) in C_E]

def build_backward(edges, input_node, output_node, n):
    """
    Greedy circuit search algorithm starting from logits.
    
    Args:
        edges: dict of dict, where edges[upstream][downstream] = score
        n: number of edges to select
        logits_node: name of the logits node (default 'lm_head')
    
    Returns:
        list of tuples (upstream, downstream, score)
    """
    C_V = {output_node}  # Circuit vertices
    C_E = set()  # Circuit edges (as tuples)
    
    for i in range(n):
        # Find candidate edges: edges not in circuit whose child is in circuit
        D = []
        for upstream, downstream, score in edges:
            edge_tuple = (upstream, downstream)
            downstream_base = get_downstream_base(downstream)
            if edge_tuple not in C_E and downstream_base in C_V:
                D.append((upstream, downstream, score))
        
        if not D:
            break  # No more edges to add
        
        # Select edge with highest absolute score
        best_edge = max(D, key=lambda x: abs(x[2]))
        upstream, downstream, score = best_edge
        upstream_base = get_upstream_base(upstream)
        downstream_base = get_downstream_base(downstream)
        
        # Add edge and parent node to circuit
        C_E.add((upstream, downstream))
        C_V.add(upstream_base)
        
    # Return edges as list of tuples with their scores
    return [(up, down, score) for up, down, score in edges if (up, down) in C_E]

def simple(edges, input_node, output_node, n):
    edges_sorted = sorted(edges, key=lambda x: abs(x[2]), reverse=True)
    top_edges = edges_sorted[:n]
    return top_edges

def random_circuit(edges, input_node, output_node, n):
    edge_count = int(1.1*n)

    while True:
        shuffled = [edges[i] for i in t.randperm(len(edges)).tolist()]
        # Phase 1: add random edges until pruned circuit has >= n edges
        found = False
        while edge_count <= len(shuffled):
            selected = shuffled[:edge_count]
            pruned = prune_circuit(selected, input_node, output_node, n)
            print(f'length of pruned with edge_count={edge_count}: {len(pruned)}')
            if len(pruned) >= n+5:
                found = True
                break
            edge_count += 100

        if not found:
            return list(pruned)

        # Phase 2: prune down toward [n-2, n+2] by removing edges one at a time
        print('pruning down')
        circuit = list(pruned)
        restart = False

        any_removed = False
        for i, edge in enumerate(list(circuit)):
            if i % 100 == 0:
                print(f'iteration {i}')
            if edge not in circuit:
                continue  # already removed by a previous reprune in this sweep
            candidate = [e for e in circuit if e != edge]
            repruned = list(prune_circuit(candidate, input_node, output_node, n))
            print(f'length of newly pruned: {len(pruned)}')
            if len(repruned) >= n - 5:
                circuit = repruned
                any_removed = True
                if len(repruned) <= n+5:
                    break
            # else: removing this edge overshoots — keep it, try next

        if not any_removed:
            # every edge removal overshoots; restart with one more edge
            edge_count += 10
            restart = True
        print(f'new edge count: {edge_count}')

        if not restart:
            return circuit


def prune_circuit(edges: list, input_node: str,
                  output_node: str, n: int) -> set:
    """
    Only include edges that are on a complete input→output path.
    """
    # Step 1: Forward pass - what's reachable from input?
    reachable_from_input = {input_node}
    forward_edges = set()
    
    changed = True
    while changed:
        changed = False
        for up, down, edge_score in edges:
            up_base = get_upstream_base(up)
            if up_base not in reachable_from_input:
                continue
            down_base = get_downstream_base(down)
            if (up, down, edge_score) not in forward_edges:
                if type(down_base) == str:
                    reachable_from_input.add(down_base)
                if type(down_base) == list:
                    reachable_from_input.update(down_base)
                forward_edges.add((up, down, edge_score))
                changed = True
    
    # Step 2: Backward pass - what can reach output?
    can_reach_output = {output_node}
    backward_edges = set()
    
    changed = True
    while changed:
        changed = False
        for up, down, edge_score in edges:
            up_base = get_upstream_base(up)
            down_base = get_downstream_base(down)
            if type(down_base) == str:
                if down_base in can_reach_output and (up, down, edge_score) not in backward_edges:
                    can_reach_output.add(up_base)
                    backward_edges.add((up, down, edge_score))
                    changed = True
            elif type(down_base) == list:
                for db in down_base:
                    if db in can_reach_output and (up, down, edge_score) not in backward_edges:
                        can_reach_output.add(up_base)
                        backward_edges.add((up, down, edge_score))
                        changed = True
    
    # Step 3: Intersection - edges on complete paths
    circuit_edges = forward_edges & backward_edges
    
    return circuit_edges

def get_node_list(nodes: dict):
    nodes_list = []
    for key in nodes.keys():
        if key == 'y': 
            continue
        nodes_list.append((key, nodes[key].sum().item()))

    return nodes_list


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
    parser.add_argument('--n', type=int, default=50)
    parser.add_argument('--method', type=str, default='simple', choices=['backward', 'forward', 'simple', 'random'])
    parser.add_argument('--plot', action="store_true", default=False)
    parser.add_argument('--unpruned', action="store_true", default=False)
    parser.add_argument('--label_vals', action="store_true", default=False)
    parser.add_argument('--granular', action="store_true", default=False)
    args = parser.parse_args()
    exp_name = args.exp_name
    GRANULAR = args.granular


    n_layers = {
        "google/gemma-2b-it": 18,
        "google/gemma-2-2b-it": 26,
        "google/gemma-2-9b-it": 42,
        "meta-llama/Llama-3.1-8B-Instruct": 32,
        "meta-llama/Llama-3.2-3B-Instruct": 28,
        "Qwen/Qwen3-8B": 36
    }[args.model_path]
    head_kv_group_ratio = {
        "google/gemma-2-2b-it": 8//4,
        "meta-llama/Llama-3.2-3B-Instruct": 24//8,
        "Qwen/Qwen3-8B": 32//8,
    }[args.model_path]
    circuit_method_fn = {
        "forward": build_forward,
        "backward": build_backward,
        "simple": simple,
        "random": random_circuit,
    }[args.method]

    save_dir1 = f"circuits/{args.model_path.split('/')[-1]}/{exp_name}"
    save_dir2 = f"/fs/nexus-scratch/scheng03/steer-interp-results/{save_dir1}"
    if os.path.exists(f"{save_dir1}"):
        save_dir = save_dir1
    elif os.path.exists(f"{save_dir2}"):
        save_dir = save_dir2
    else:
        print(f"save dirs {save_dir1}, {save_dir2} do not exist")
        

    with open(f"{save_dir}/patching_results.pt", "rb") as datafile:
        data = t.load(datafile, map_location=t.device("cpu"))

    nodes = data['nodes']
    edges = data['edges']
    total_poi = data['num_positions']
    print(f'total number of positions evaluated: {total_poi}')

    sanity_check_values(nodes, edges)
    inspect_nodes(nodes)
    inspect_edges(edges)

    edges_list, up_names, down_names = get_edge_list(edges)

    print(f"Total number of edges: {len(edges_list)}. Asked for fraction: {args.n / len(edges_list)}")
    # print('=== Down Names ===')
    # print(len(down_names))
    # print('=== Up Names ===')
    # print(len(up_names))

    input_node_name = list(nodes.keys())[1]
    output_node_name = list(nodes.keys())[-1]

    edge_count = args.n
    while True:
        unpruned_circuit_edges = circuit_method_fn(edges_list, input_node_name, output_node_name, edge_count)
        if args.unpruned:
            pruned_circuit_edges = unpruned_circuit_edges
        else:
            pruned_circuit_edges = prune_circuit(unpruned_circuit_edges, input_node_name, output_node_name, args.n)
        if len(pruned_circuit_edges) < args.n:
            edge_count += 1
        else:
            break
    print(f'Method: {args.method}. Asked for minimum {args.n} edges, obtained {len(pruned_circuit_edges)} with {edge_count} starting edges')

    # inspect_circuit(pruned_circuit_edges, up_names, down_names, edges_list)

    edge_name = f"edges_asked{args.n}_{args.method}_actual{len(pruned_circuit_edges)}_{'unpruned' if args.unpruned else 'pruned'}"
    with open(os.path.join(save_dir, edge_name+".json"), 'w') as edgesfile:
        edges_d = []
        for (up, down, score) in pruned_circuit_edges:
            edges_d.append({
                "up": up, "down": down, "score": score
            })
        json.dump(edges_d, edgesfile, indent=4)

    if args.plot:
        graph_name = f"graph_asked{args.n}_{args.method}_actual{len(pruned_circuit_edges)}_{'unpruned' if args.unpruned else 'pruned'}"
        if GRANULAR:
            plot_module_circuit_granular(
                nodes,
                pruned_circuit_edges,
                n_layers,
                head_kv_group_ratio,
                save_path=os.path.join(save_dir, graph_name),
                label_vals=args.label_vals
            )
        else:
            plot_module_circuit(
                nodes,
                pruned_circuit_edges,
                n_layers,
                save_path=os.path.join(save_dir, graph_name),
                label_vals=args.label_vals
            )
