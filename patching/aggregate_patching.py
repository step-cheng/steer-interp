import torch as t
from collections import defaultdict


def aggregate_patching_results(results: list[dict], weighted: bool=False) -> dict:
    """
    Aggregate patching results from multiple patching_results.pt files.
    Weights each file's nodes/edges by its num_positions.

    Args:
        result_paths: list of paths to patching_results.pt files

    Returns:
        dict with aggregated 'nodes', 'edges', 'num_positions', and 'examples'
    """
    total_num_pos = sum(r["num_positions"] for r in results)
    n = len(results)

    # aggregate nodes
    running_nodes = None
    for r in results:
        w = 1 if weighted else r["num_positions"]
        if running_nodes is None:
            running_nodes = {k: w * v for k, v in r["nodes"].items()}
        else:
            for k, v in r["nodes"].items():
                running_nodes[k] = running_nodes.get(k, t.zeros_like(v)) + w * v

    node_denom = n if weighted else total_num_pos
    nodes = {k: v / node_denom for k, v in running_nodes.items()}

    # aggregate edges (may be None if nodes_only)
    edges = None
    if any(r.get("edges") is not None for r in results):
        running_edges = defaultdict(dict)
        results_with_edges = [r for r in results if r.get("edges") is not None]
        edge_denom = len(results_with_edges) if weighted else sum(r["num_positions"] for r in results_with_edges)
        for r in results_with_edges:
            w = 1 if weighted else r["num_positions"]
            for up, downstream in r["edges"].items():
                for down, val in downstream.items():
                    if down in running_edges[up]:
                        running_edges[up][down] = running_edges[up][down] + w * val
                    else:
                        running_edges[up][down] = w * val
        edges = {
            up: {down: v / edge_denom for down, v in downstream.items()}
            for up, downstream in running_edges.items()
        }

    examples = [e for r in results for e in r.get("examples", [])]

    return {
        "nodes": nodes,
        "edges": edges,
        "num_positions": total_num_pos,
        "examples": examples,
    }


if __name__ == "__main__":
    import argparse, os, json

    parser = argparse.ArgumentParser()
    parser.add_argument("result_names", nargs="+", help="paths to patching_results.pt")
    parser.add_argument("--model_path", required=True, help="model name")
    parser.add_argument("--save_name", required=True, help="output .pt file")
    parser.add_argument("--weighted", action='store_true', help="equally divides effect across datasets")
    args = parser.parse_args()

    circuit_dir1 = f"circuits/{args.model_path.split('/')[-1]}"
    circuit_dir2 = f"/fs/nexus-scratch/scheng03/steer-interp-results/{circuit_dir1}"
    if os.path.exists(os.path.join(circuit_dir1, args.result_names[0], 'patching_results.pt')):
        circuits_dir = circuit_dir1
    elif os.path.exists(os.path.join(circuit_dir2, args.result_names[0], 'patching_results.pt')):
        circuits_dir = circuit_dir2
    else:

        print(f"save dirs {os.path.join(circuit_dir1, args.result_names[0], 'patching_results.pt')}, {os.path.join(circuit_dir2, args.result_names[0], 'patching_results.pt')} do not exist")

    print('loading results')
    results = [t.load(os.path.join(circuits_dir, p, 'patching_results.pt'), map_location="cpu") for p in args.result_names]

    print('aggregating results...')
    aggregated = aggregate_patching_results(results, args.weighted)

    os.makedirs(os.path.join(circuits_dir, args.save_name), exist_ok=True)
    with open(os.path.join(circuits_dir, args.save_name, 'patching_results.pt'), "wb") as f:
        t.save(aggregated, f)
    
    with open(os.path.join(circuits_dir, args.save_name, 'agg_names.json'), "w") as f:
        json.dump({
            'names': [os.path.join(circuits_dir, p, 'patching_results.pt') for p in args.result_names]
        }, f, indent=4)
    
    
    print(f"Saved aggregated results ({aggregated['num_positions']} positions) to {args.save_name}")
