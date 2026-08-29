import torch as t
from utils.plotting_utils import plot_position_circuit_compact, plot_position_circuit
import json
import os
from config_act_patch import Config
import argparse

def sanity_check_values(nodes, edges):
    for k, v in nodes.items():
        assert t.isnan(v).sum() == 0, f"nan at {k}"
    for up in edges:
        for down in edges[up]:
            assert t.isnan(edges[up][down]).sum() == 0, f"nan at edge {up} to {down}"
    print('passed nan sanity check')

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
        for upstream, downstream, pos, score in edges:
            edge_tuple = (upstream, downstream)
            upstream_base = "_".join(upstream.split("_")[:2])
            if edge_tuple not in C_E and upstream_base in C_V:
                D.append((upstream, downstream, score))
        
        if not D:
            break  # No more edges to add
        
        # Select edge with highest absolute score
        best_edge = max(D, key=lambda x: abs(x[3]))
        upstream, downstream, score = best_edge
        upstream_base = "_".join(upstream.split("_")[:2])
        downstream_base = "_".join(downstream.split("_")[:2])
        
        # Add edge and parent node to circuit
        C_E.add((upstream, downstream))
        C_V.add(downstream_base)
        
    # Return edges as list of tuples with their scores
    return [(up, down, pos, score) for up, down, pos, score in edges if (up, down) in C_E]

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
        for upstream, downstream, pos, score in edges:
            edge_tuple = (upstream, downstream)
            upstream_base = "_".join(upstream.split("_")[:2])
            if edge_tuple not in C_E and upstream_base in C_V:
                D.append((upstream, downstream, score))
        
        if not D:
            print('broke')
            break  # No more edges to add
        
        # Select edge with highest absolute score
        best_edge = max(D, key=lambda x: abs(x[3]))
        upstream, downstream, score = best_edge
        upstream_base = "_".join(upstream.split("_")[:2])
        downstream_base = "_".join(downstream.split("_")[:2])
        
        # Add edge and parent node to circuit
        C_E.add((upstream, downstream))
        C_V.add(downstream_base)
        
    print(C_V)
    # Return edges as list of tuples with their scores
    return [(up, down, pos, score) for up, down, pos, score in edges if (up, down) in C_E]

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
        for upstream, downstream, pos, score in edges:
            edge_tuple = (upstream, downstream)
            downstream_base = "_".join(downstream.split("_")[:2])
            if edge_tuple not in C_E and downstream_base in C_V:
                D.append((upstream, downstream, score))
        
        if not D:
            break  # No more edges to add
        
        # Select edge with highest absolute score
        best_edge = max(D, key=lambda x: abs(x[3]))
        upstream, downstream, score = best_edge
        upstream_base = "_".join(upstream.split("_")[:2])
        downstream_base = "_".join(downstream.split("_")[:2])
        
        # Add edge and parent node to circuit
        C_E.add((upstream, downstream))
        C_V.add(upstream_base)
        
    # Return edges as list of tuples with their scores
    return [(up, down, pos, score) for up, down, pos, score in edges if (up, down) in C_E]

def simple(edges, input_node, output_node, n):
    edges_sorted = sorted(edges, key=lambda x: abs(x[3]), reverse=True)
    top_edges = edges_sorted[:n]
    return top_edges


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
        for up, down, pos, edge_score in edges:
            up_base = "_".join(up.split("_")[:2]) if up.startswith("attn") else up
            if up_base not in reachable_from_input:
                continue
            down_base = "_".join(down.split("_")[:2]) if down.startswith("attn") else down
            if (up, down, pos, edge_score) not in forward_edges:
                reachable_from_input.add(down_base)
                forward_edges.add((up, down, pos, edge_score))
                changed = True
    
    # Step 2: Backward pass - what can reach output?
    can_reach_output = {output_node}
    backward_edges = set()
    
    changed = True
    while changed:
        changed = False
        for up, down, pos, edge_score in edges:
            up_base = "_".join(up.split("_")[:2]) if up.startswith("attn") else up
            down_base = "_".join(down.split("_")[:2]) if down.startswith("attn") else down
            if down_base in can_reach_output and (up, down, pos, edge_score) not in backward_edges:
                can_reach_output.add(up_base)
                backward_edges.add((up, down, pos, edge_score))
                changed = True
    
    # Step 3: Intersection - edges on complete paths
    circuit_edges = forward_edges & backward_edges
    
    return circuit_edges

def no_constraint(edges, input_node, output_node):
    return edges

def get_edge_list(edges: dict):
    edges_listed = []
    for up in edges:
        for down in edges[up]:
            edge_scores = edges[up][down]
            num_pos = edge_scores.shape[0]
            for pos in range(num_pos):
                edges_listed.append((up, down, pos, edges[up][down][pos].sum().item()))
    return edges_listed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--model_path', type=str, default='google/gemma-2-2b-it')
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--method', type=str, default='simple', choices=['backward', 'forward', 'simple'])
    parser.add_argument('--unpruned', action="store_true", default=False)
    parser.add_argument('--plot', action="store_true", default=False)
    args = parser.parse_args()
    # exp_name = 'gemma2-2b_base_harmful_5thresh'
    exp_name = args.exp_name

    save_dir = f"circuits_pos/{args.model_path.split('/')[-1]}/{exp_name}"

    n_layers = {
        "google/gemma-2b-it": 18,
        "google/gemma-2-2b-it": 26,
        "google/gemma-2-9b-it": 42,
        "meta-llama/Llama-3.1-8B-Instruct": 32
    }[args.model_path]
    circuit_method_fn = {
        "forward": build_forward,
        "backward": build_backward,
        "simple": simple,
    }[args.method]

    with open(f"{save_dir}/config.json", "r") as configfile:
        loaded_dict = json.load(configfile)
    config = Config.from_dict(loaded_dict)

    with open(f"{save_dir}/patching_results.pt", "rb") as datafile:
        data = t.load(datafile, map_location=t.device("cpu"))

    nodes_list = data['nodes']
    edges_list = data['edges']
    example_list = data['examples']
    decoded_list = data['examples_decoded']

    example_idx = 0
    example = example_list[example_idx]
    decoded = decoded_list[example_idx]
    print(decoded)

    nodes = nodes_list[example_idx]
    edges = edges_list[example_idx]

    sanity_check_values(nodes, edges)

    edges_list = get_edge_list(edges)
    print(f"mlp 25 pos 26: {edges['mlp_25']['lm_head'][-1]}, {nodes['mlp_25'].shape}")
    print(f"Total number of edges: {len(edges_list)}. Asked for fraction: {args.n / len(edges_list)}")

    input_node_name = list(nodes.keys())[0]
    output_node_name = list(nodes.keys())[-2]
    print(f"input node: {input_node_name}, output node: {output_node_name}")

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

    # neg_circuit_f_name = 'neg_circuit_components.json'
    # with open(os.path.join(circuits_dir, f"{name}_{neg_circuit_f_name}"), 'w') as f:
    #     json.dump(neg_data, f, indent=4)
    edge_name = f"edges_asked{args.n}_{args.method}_actual{len(pruned_circuit_edges)}"
    with open(os.path.join(save_dir, edge_name+".json"), 'w') as edgesfile:
        edges_d = []
        for (up, down, pos, score) in pruned_circuit_edges:
            edges_d.append({
                "up": up, "down": down, "score": score
            })
        json.dump(edges_d, edgesfile, indent=4)

    if args.plot:
        graph_name = f"graph_asked{args.n}_{args.method}_actual{len(pruned_circuit_edges)}"
        plot_position_circuit(
            nodes,
            pruned_circuit_edges,
            n_layers,
            decoded,
            save_path=os.path.join(save_dir, graph_name)
        )
