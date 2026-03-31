import os
from collections import defaultdict
from graphviz import Digraph
import math


def plot_module_circuit(
    nodes,
    edges: set,
    layers,
    pen_thickness=3.0,
    save_path="circuit",
    label_vals=False,
):
    """Plot a circuit at module-level granularity."""
    import torch
    
    # Convert nodes to scalar values (sum across all dimensions)
    nodes_with_edges = set()
    for upstream_name, downstream_name, edge_score in edges:
        nodes_with_edges.add(upstream_name)
        nodes_with_edges.add(downstream_name)

    node_values = {}
    for name, tensor in nodes.items():
        if name not in nodes_with_edges:
            continue
        if isinstance(tensor, torch.Tensor):
            node_values[name] = tensor.sum().item()
        else:
            node_values[name] = float(tensor)
    
    def value_to_color(value):
        if not node_values:
            return "#FFFFFF", "#000000"
        
        scale = max(
            abs(min(node_values.values())),
            abs(max(node_values.values()))
        )
        if scale == 0:
            return "#FFFFFF", "#000000"
        
        normalized = value / scale
        
        if normalized < 0:
            red = 255
            green = blue = int((1 + normalized) * 255)
        elif normalized > 0:
            blue = 255
            red = green = int((1 - normalized) * 255)
        else:
            red = green = blue = 255
        
        brightness = red * 0.299 + green * 0.587 + blue * 0.114
        text_color = "#000000" if brightness > 170 else "#FFFFFF"
        fill_color = f"#{red:02X}{green:02X}{blue:02X}"
        
        return fill_color, text_color
    
    # Identify QKV endpoints that have edges
    qkv_nodes_to_add = {}
    qkv_to_heads = defaultdict(set)
    
    for upstream_name, downstream_name, edge_val in edges:
        if '_q' in downstream_name or '_k' in downstream_name or '_v' in downstream_name:
            parts = downstream_name.split('_')
            if parts[0] == 'attn' and parts[2] in ['q', 'k', 'v']:
                if downstream_name not in qkv_nodes_to_add:
                    qkv_nodes_to_add[downstream_name] = 0
    
    # Find attention head nodes and map QKV to heads
    head_nodes = {}
    for name, value in node_values.items():
        if name.startswith('attn_') and '_h' in name:
            parts = name.split('_')
            if parts[0] == 'attn' and parts[2].startswith('h'):
                layer = int(parts[1])
                head_nodes[name] = layer
                for qkv_suffix in ['q', 'k', 'v']:
                    qkv_name = f"attn_{layer}_{qkv_suffix}"
                    qkv_to_heads[qkv_name].add(name)
    
    # Create graph
    G = Digraph(name="Module Circuit")
    G.graph_attr.update(
        rankdir="BT",
        ranksep="0.5",
        nodesep="0.3",
        newrank="true",  # Enable newrank for better rank handling
    )
    G.node_attr.update(shape="box", style="rounded,filled")
    
    # Helper to parse layer from name
    def get_layer(name):
        if name == "embed":
            return -1
        elif name.startswith("lm_head"):
            return layers + 1
        elif name.startswith("resid_"):
            return int(name.split("_")[1]) + 0.5  # Put resid between layers
        elif name.startswith("attn_") and '_h' in name:
            return int(name.split("_")[1])
        elif name.startswith("mlp_"):
            return int(name.split("_")[1])
        else:
            raise ValueError(f'unknown component: {name}')
    
    def get_qkv_layer(name):
        # QKV nodes go slightly below their layer's attention heads
        parts = name.split('_')
        return int(parts[1]) - 0.1
    
    # Group nodes by layer
    nodes_by_layer = defaultdict(list)
    qkv_by_layer = defaultdict(list)
    included_nodes = set()
    
    # Add regular nodes
    for name, value in node_values.items():        
        layer = get_layer(name)
        fill_color, text_color = value_to_color(value)
        label = f"{name}\\n{value:.4f}" if label_vals else f"{name}"
        
        G.node(
            name,
            label=label,
            fillcolor=fill_color,
            fontcolor=text_color,
        )
        
        nodes_by_layer[layer].append(name)
        included_nodes.add(name)
    
    # Add QKV nodes
    for qkv_name, _ in qkv_nodes_to_add.items():
        layer = get_qkv_layer(qkv_name)
        
        G.node(
            qkv_name,
            label=f"{qkv_name}",
            fillcolor="#E0E0E0",
            fontcolor="#000000",
            shape="ellipse",
        )
        
        qkv_by_layer[layer].append(qkv_name)
        included_nodes.add(qkv_name)
    
    # Combine all layers and sort
    all_layers = sorted(set(nodes_by_layer.keys()) | set(qkv_by_layer.keys()))
    
    # Create subgraphs with rank='same' for each layer
    for layer in all_layers:
        with G.subgraph() as s:
            s.attr(rank='same')
            for node in nodes_by_layer.get(layer, []):
                s.node(node)
            for node in qkv_by_layer.get(layer, []):
                s.node(node)
    
    # Add invisible edges between consecutive layers to enforce ordering
    # Pick one representative node from each layer
    layer_representatives = {}
    for layer in all_layers:
        if nodes_by_layer.get(layer):
            layer_representatives[layer] = nodes_by_layer[layer][0]
        elif qkv_by_layer.get(layer):
            layer_representatives[layer] = qkv_by_layer[layer][0]
    
    sorted_layers = sorted(layer_representatives.keys())
    for i in range(len(sorted_layers) - 1):
        lower_layer = sorted_layers[i]
        upper_layer = sorted_layers[i + 1]
        lower_node = layer_representatives[lower_layer]
        upper_node = layer_representatives[upper_layer]
        
        # Add invisible edge to enforce ordering
        G.edge(
            lower_node,
            upper_node,
            style="invis",
            weight="10",  # High weight to prioritize this constraint
        )
    
    # Collect edge magnitudes for normalization
    edge_magnitudes = [abs(e[2]) for e in edges]
    max_edge = max(edge_magnitudes) if edge_magnitudes else 1.0
    min_edge = min(edge_magnitudes) if edge_magnitudes else 0.0
    
    # Add visible edges
    for upstream_name, downstream_name, edge_score in edges:                
        if max_edge > min_edge:
            normalized = (abs(edge_score) - min_edge) / (max_edge - min_edge)
            thickness = 0.5 + normalized * pen_thickness
        else:
            thickness = pen_thickness / 2
        
        color = "red" if edge_score < 0 else "blue"
        G.edge(
            upstream_name,
            downstream_name,
            penwidth=f"{thickness:.2f}",
            color=color,
            label=f"{edge_score:.3f}" if label_vals else None,
            fontsize="10",
        )
    
    # Add grey edges from QKV nodes to their corresponding attention heads
    for qkv_name, head_names in qkv_to_heads.items():
        if qkv_name not in included_nodes:
            continue
        for head_name in head_names:
            if head_name in included_nodes:
                G.edge(
                    qkv_name,
                    head_name,
                    color="gray",
                    style="dashed",
                    penwidth="1.0",
                    arrowsize="0.5",
                )
    
    # Render
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    G.render(save_path, format="pdf", cleanup=True)
    print(f"Circuit visualization saved to {save_path}.pdf")



import os
from collections import defaultdict
from graphviz import Digraph


def plot_position_circuit(
    nodes: dict,  
    edges: list,  # [(upstream, downstream, pos, score)]
    layers: int,
    decoded: int,
    seq_len: int = 16,
    pen_thickness: float = 3.0,
    save_path: str = "circuit_position",
    show_position_labels: bool = True,
    token_labels: list = None,  # Optional list of token strings for labeling
):
    """
    Plot a circuit at position-level granularity.
    
    Each node represents a (module, position) pair.
    Positions are arranged left-to-right within each layer.
    
    Args:
        nodes: Node scores. Can be:
            - {(module_name, pos): score}
            - {module_name: {pos: score}}
        edges: Edge list as [(upstream, downstream, pos, score)]
            Edges connect modules at the same position (activation patching
            operates on the residual stream at each position independently).
        layers: Number of layers in the model
        seq_len: Sequence length
        pen_thickness: Maximum edge thickness
        save_path: Path to save the visualization
        show_position_labels: Whether to show position indices on nodes
        token_labels: Optional token strings for position labels
    """
    try:
        import torch
        has_torch = True
    except ImportError:
        has_torch = False
    
    def to_float(value):
        if has_torch and isinstance(value, torch.Tensor):
            return value.sum().item()
        return float(value)
    
    # Normalize nodes to {(module, pos): score} format
    node_scores = {}
    for module, scores in nodes.items():
        if module == 'y': continue
        for pos in range(len(scores)):
            node_scores[(module, pos)] = to_float(scores[pos])
    
    # Normalize edges to [(upstream_module, downstream_module, pos, score)]
    normalized_edges = []
    for edge in edges:
        if len(edge) != 4:
            raise ValueError(f"Edge must have 4 elements (upstream, downstream, pos, score), got {len(edge)}: {edge}")
        upstream, downstream, pos, score = edge
        normalized_edges.append((upstream, downstream, pos, to_float(score)))
    
    # Collect all nodes that have edges
    nodes_with_edges = set()
    for upstream, downstream, pos, _ in normalized_edges:
        nodes_with_edges.add((upstream, pos))
        nodes_with_edges.add((downstream, pos))
    
    # If no node scores provided, infer from edges
    if not node_scores:
        for node in nodes_with_edges:
            node_scores[node] = 0.0
    
    # Filter to only nodes with edges
    node_scores = {k: v for k, v in node_scores.items() if k in nodes_with_edges}
    
    def value_to_color(value):
        if not node_scores:
            return "#FFFFFF", "#000000"
        
        all_values = list(node_scores.values())
        scale = max(abs(min(all_values)), abs(max(all_values)))
        if scale == 0:
            return "#FFFFFF", "#000000"
        
        normalized = value / scale
        
        if normalized < 0:
            red = 255
            green = blue = int((1 + normalized) * 255)
        elif normalized > 0:
            blue = 255
            red = green = int((1 - normalized) * 255)
        else:
            red = green = blue = 255
        
        brightness = red * 0.299 + green * 0.587 + blue * 0.114
        text_color = "#000000" if brightness > 170 else "#FFFFFF"
        fill_color = f"#{red:02X}{green:02X}{blue:02X}"
        
        return fill_color, text_color
    
    def get_layer(module_name):
        """Get layer number for vertical positioning."""
        if module_name == "embed":
            return -1
        elif module_name.startswith("lm_head"):
            return layers + 1
        elif module_name.startswith("resid_"):
            return int(module_name.split("_")[1]) + 0.5
        elif module_name.startswith("attn_") and '_h' in module_name:
            return int(module_name.split("_")[1])
        elif module_name.startswith("attn_"):
            # Handle attn_L_q, attn_L_k, attn_L_v
            parts = module_name.split("_")
            return int(parts[1]) - 0.1
        elif module_name.startswith("mlp_"):
            return int(module_name.split("_")[1])
        else:
            # Try to extract layer number
            import re
            match = re.search(r'_(\d+)', module_name)
            if match:
                return int(match.group(1))
            raise ValueError(f'Unknown component: {module_name}')
    
    def node_id(module, pos):
        """Create unique node ID for (module, position) pair."""
        return f"{module}_p{pos}"
    
    # Create graph
    G = Digraph(name="Position-Level Circuit")
    G.graph_attr.update(
        rankdir="BT",  # Bottom to top
        ranksep="1.0",
        nodesep="0.15",  # Tighter horizontal spacing
        newrank="true",
    )
    G.node_attr.update(shape="box", style="rounded,filled", fontsize="9")
    
    # Group nodes by layer
    nodes_by_layer = defaultdict(list)
    
    for (module, pos), score in node_scores.items():
        layer = get_layer(module)
        nodes_by_layer[layer].append((module, pos, score))
    
    # Sort nodes within each layer: first by module name, then by position
    for layer in nodes_by_layer:
        # Sort by position primarily (left to right), then by module name
        nodes_by_layer[layer].sort(key=lambda x: (x[1], x[0]))
    
    # Add nodes with their visual properties
    for layer, layer_nodes in nodes_by_layer.items():
        for module, pos, score in layer_nodes:
            nid = node_id(module, pos)
            fill_color, text_color = value_to_color(score)
            
            # Create label
            if show_position_labels:
                if token_labels and pos < len(token_labels):
                    pos_label = f"'{token_labels[pos]}'"
                else:
                    pos_label = f"p{pos}"
                label = f"{module}\\n{pos_label}\\n{score:.4f}"
            else:
                label = f"{module}\\n{score:.4f}"
            if module.startswith("resid"):
                label += f"{decoded[pos]}"
            
            G.node(
                nid,
                label=label,
                fillcolor=fill_color,
                fontcolor=text_color,
            )
    
    # Create subgraphs with rank='same' for each layer
    # Within each layer, we need to enforce left-to-right ordering by position
    all_layers = sorted(nodes_by_layer.keys())
    
    for layer in all_layers:
        layer_nodes = nodes_by_layer[layer]
        
        with G.subgraph() as s:
            s.attr(rank='same')
            for module, pos, _ in layer_nodes:
                s.node(node_id(module, pos))
        
        # Add invisible edges to enforce left-to-right position ordering within layer
        # Group by position first
        by_position = defaultdict(list)
        for module, pos, _ in layer_nodes:
            by_position[pos].append((module, pos))
        
        sorted_positions = sorted(by_position.keys())
        for i in range(len(sorted_positions) - 1):
            left_pos = sorted_positions[i]
            right_pos = sorted_positions[i + 1]
            
            # Connect last node of left position to first node of right position
            left_nodes = by_position[left_pos]
            right_nodes = by_position[right_pos]
            
            # Add invisible edge from rightmost of left_pos to leftmost of right_pos
            G.edge(
                node_id(left_nodes[-1][0], left_nodes[-1][1]),
                node_id(right_nodes[0][0], right_nodes[0][1]),
                style="invis",
                weight="100",
            )
    
    # Add invisible edges between layers for vertical ordering
    layer_representatives = {}
    for layer in all_layers:
        if nodes_by_layer[layer]:
            module, pos, _ = nodes_by_layer[layer][0]
            layer_representatives[layer] = node_id(module, pos)
    
    sorted_layers = sorted(layer_representatives.keys())
    for i in range(len(sorted_layers) - 1):
        lower_layer = sorted_layers[i]
        upper_layer = sorted_layers[i + 1]
        G.edge(
            layer_representatives[lower_layer],
            layer_representatives[upper_layer],
            style="invis",
            weight="10",
        )
    
    # Collect edge magnitudes for normalization
    edge_magnitudes = [abs(e[3]) for e in normalized_edges]
    max_edge = max(edge_magnitudes) if edge_magnitudes else 1.0
    min_edge = min(edge_magnitudes) if edge_magnitudes else 0.0
    
    # Add visible edges
    for upstream, downstream, pos, score in normalized_edges:
        if max_edge > min_edge:
            normalized = (abs(score) - min_edge) / (max_edge - min_edge)
            thickness = 0.5 + normalized * pen_thickness
        else:
            thickness = pen_thickness / 2
        
        color = "red" if score < 0 else "blue"
        
        G.edge(
            node_id(upstream, pos),
            node_id(downstream, pos),
            penwidth=f"{thickness:.2f}",
            color=color,
            label=f"{score:.3f}",
            fontsize="8",
        )
    
    # Render
    dir_path = os.path.dirname(save_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    G.render(save_path, format="png", cleanup=True)
    print(f"Circuit visualization saved to {save_path}.png")
    
    return G


def plot_position_circuit_compact(
    edges: list,  # [(upstream, downstream, pos, score)] or 5-tuple
    layers: int,
    seq_len: int = 16,
    pen_thickness: float = 3.0,
    save_path: str = "circuit_position_compact",
    token_labels: list = None,
    group_by_module: bool = True,
):
    """
    More compact visualization that groups positions within modules.
    
    Creates a grid-like layout where:
    - Rows are layers (bottom to top)
    - Within each layer, modules are separate clusters
    - Within each module, positions go left to right
    """
    try:
        import torch
        has_torch = True
    except ImportError:
        has_torch = False
    
    def to_float(value):
        if has_torch and isinstance(value, torch.Tensor):
            return value.sum().item()
        return float(value)
    
    # Normalize edges to [(upstream, downstream, pos, score)]
    normalized_edges = []
    for edge in edges:
        if len(edge) != 4:
            raise ValueError(f"Edge must have 4 elements (upstream, downstream, pos, score), got {len(edge)}: {edge}")
        upstream, downstream, pos, score = edge
        normalized_edges.append((upstream, downstream, pos, to_float(score)))
    
    # Collect all nodes
    all_nodes = set()
    for upstream, downstream, pos, _ in normalized_edges:
        all_nodes.add((upstream, pos))
        all_nodes.add((downstream, pos))
    
    def get_layer(module_name):
        if module_name == "embed":
            return -1
        elif module_name.startswith("lm_head"):
            return layers + 1
        elif module_name.startswith("resid_"):
            return int(module_name.split("_")[1]) + 0.5
        elif module_name.startswith("attn_") and '_h' in module_name:
            return int(module_name.split("_")[1])
        elif module_name.startswith("attn_"):
            parts = module_name.split("_")
            return int(parts[1]) - 0.1
        elif module_name.startswith("mlp_"):
            return int(module_name.split("_")[1])
        else:
            import re
            match = re.search(r'_(\d+)', module_name)
            if match:
                return int(match.group(1))
            return 0
    
    def node_id(module, pos):
        return f"{module}_p{pos}"
    
    G = Digraph(name="Position-Level Circuit (Compact)")
    G.graph_attr.update(
        rankdir="BT",
        ranksep="0.8",
        nodesep="0.1",
        newrank="true",
        compound="true",  # Allow edges to clusters
    )
    G.node_attr.update(shape="box", style="rounded,filled", fontsize="8", width="0.3", height="0.3")
    
    # Group nodes by (layer, module)
    nodes_by_layer_module = defaultdict(lambda: defaultdict(list))
    for module, pos in all_nodes:
        layer = get_layer(module)
        nodes_by_layer_module[layer][module].append(pos)
    
    # Sort positions within each module
    for layer in nodes_by_layer_module:
        for module in nodes_by_layer_module[layer]:
            nodes_by_layer_module[layer][module].sort()
    
    # Create subgraphs for each layer
    all_layers = sorted(nodes_by_layer_module.keys())
    
    for layer in all_layers:
        with G.subgraph(name=f"cluster_layer_{layer}") as layer_subgraph:
            layer_subgraph.attr(rank="same", style="invis")
            
            # Sort modules within layer for consistent ordering
            sorted_modules = sorted(nodes_by_layer_module[layer].keys())
            
            for module in sorted_modules:
                positions = nodes_by_layer_module[layer][module]
                
                if group_by_module:
                    # Create a cluster for each module
                    with layer_subgraph.subgraph(name=f"cluster_{module}") as mod_subgraph:
                        mod_subgraph.attr(label=module, style="rounded", color="gray")
                        
                        for pos in positions:
                            nid = node_id(module, pos)
                            if token_labels and pos < len(token_labels):
                                label = f"{pos}:'{token_labels[pos][:3]}'"
                            else:
                                label = f"p{pos}"
                            mod_subgraph.node(nid, label=label, fillcolor="#E8E8E8")
                        
                        # Invisible edges within module for ordering
                        for i in range(len(positions) - 1):
                            mod_subgraph.edge(
                                node_id(module, positions[i]),
                                node_id(module, positions[i + 1]),
                                style="invis",
                                weight="100",
                            )
                else:
                    for pos in positions:
                        nid = node_id(module, pos)
                        if token_labels and pos < len(token_labels):
                            label = f"{module}\\np{pos}:'{token_labels[pos][:4]}'"
                        else:
                            label = f"{module}\\np{pos}"
                        layer_subgraph.node(nid, label=label, fillcolor="#E8E8E8")
    
    # Add edges between layers for ordering
    for i in range(len(all_layers) - 1):
        lower = all_layers[i]
        upper = all_layers[i + 1]
        
        lower_modules = list(nodes_by_layer_module[lower].keys())
        upper_modules = list(nodes_by_layer_module[upper].keys())
        
        if lower_modules and upper_modules:
            lower_mod = lower_modules[0]
            upper_mod = upper_modules[0]
            lower_pos = nodes_by_layer_module[lower][lower_mod][0]
            upper_pos = nodes_by_layer_module[upper][upper_mod][0]
            
            G.edge(
                node_id(lower_mod, lower_pos),
                node_id(upper_mod, upper_pos),
                style="invis",
                weight="10",
            )
    
    # Edge visualization
    edge_magnitudes = [abs(e[3]) for e in normalized_edges]
    max_edge = max(edge_magnitudes) if edge_magnitudes else 1.0
    min_edge = min(edge_magnitudes) if edge_magnitudes else 0.0
    
    for upstream, downstream, pos, score in normalized_edges:
        if max_edge > min_edge:
            normalized = (abs(score) - min_edge) / (max_edge - min_edge)
            thickness = 0.5 + normalized * pen_thickness
        else:
            thickness = pen_thickness / 2
        
        color = "red" if score < 0 else "blue"
        
        G.edge(
            node_id(upstream, pos),
            node_id(downstream, pos),
            penwidth=f"{thickness:.2f}",
            color=color,
            label=f"{score:.2f}",
            fontsize="7",
        )
    
    dir_path = os.path.dirname(save_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    G.render(save_path, format="png", cleanup=True)
    print(f"Compact circuit visualization saved to {save_path}.png")
    
    return G


# Example usage
if __name__ == "__main__":
    # Example edges: (upstream_module, downstream_module, position, score)
    # Edges connect modules at the same position - this is how activation
    # patching works (interventions propagate through the residual stream
    # at each position independently)
    example_edges = [
        ("embed", "attn_0_h0", 0, 0.5),
        ("embed", "attn_0_h0", 5, 0.3),
        ("embed", "attn_0_h1", 10, 0.4),
        ("attn_0_h0", "mlp_0", 0, 0.6),
        ("attn_0_h0", "mlp_0", 5, -0.2),
        ("attn_0_h1", "mlp_0", 10, 0.35),
        ("mlp_0", "resid_0", 0, 0.7),
        ("mlp_0", "resid_0", 5, 0.25),
        ("mlp_0", "resid_0", 10, 0.4),
        ("resid_0", "attn_1_h0", 5, 0.8),
        ("resid_0", "attn_1_h0", 10, 0.3),
        ("attn_1_h0", "mlp_1", 5, -0.4),
        ("attn_1_h0", "mlp_1", 10, 0.5),
        ("mlp_1", "lm_head", 15, 0.9),
    ]
    
    # Optional: node scores
    example_nodes = {
        ("embed", 0): 0.1,
        ("embed", 5): 0.15,
        ("embed", 10): 0.12,
        ("attn_0_h0", 0): 0.3,
        ("attn_0_h0", 5): 0.25,
        ("attn_0_h1", 10): 0.28,
        ("mlp_0", 0): 0.4,
        ("mlp_0", 5): -0.1,
        ("mlp_0", 10): 0.35,
        ("resid_0", 0): 0.5,
        ("resid_0", 5): 0.45,
        ("resid_0", 10): 0.42,
        ("attn_1_h0", 5): 0.6,
        ("attn_1_h0", 10): 0.55,
        ("mlp_1", 5): -0.2,
        ("mlp_1", 10): 0.65,
        ("lm_head", 15): 0.9,
    }
    
    token_labels = ["The", "quick", "brown", "fox", "jumps", "over", 
                    "the", "lazy", "dog", ".", "A", "cat", "sleeps", 
                    ".", "End", "!"]
    
    # Plot position-level circuit
    plot_position_circuit(
        nodes=example_nodes,
        edges=example_edges,
        layers=2,
        seq_len=16,
        save_path="example_position_circuit",
        token_labels=token_labels,
    )
    
    # Compact version (groups positions within module clusters)
    plot_position_circuit_compact(
        edges=example_edges,
        layers=2,
        seq_len=16,
        save_path="example_compact_circuit",
        token_labels=token_labels,
    )