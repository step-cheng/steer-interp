import torch
import random
import json
import os
import argparse

from refusal_direction.dataset.load_dataset import load_dataset_split, load_dataset

from refusal_direction.pipeline.config import Config
from refusal_direction.pipeline.model_utils.model_factory import construct_model_base
from refusal_direction.pipeline.utils.hook_utils import get_activation_addition_input_pre_hook, get_all_direction_ablation_hooks

from refusal_direction.pipeline.submodules.generate_directions import generate_directions
from refusal_direction.pipeline.submodules.select_direction import select_direction, get_refusal_scores
from refusal_direction.pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak
from refusal_direction.pipeline.submodules.evaluate_loss import evaluate_loss

def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--vector_path', type=str, default=None)
    parser.add_argument('--completion_suffix', type=str, default=None)
    parser.add_argument('--no_eval', action='store_true', default=False)
    parser.add_argument('--no_generate', action='store_true', default=False)
    return parser.parse_args()

def load_and_sample_datasets(cfg):
    """
    Load datasets and sample them based on the configuration.

    Returns:
        Tuple of datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """
    random.seed(42)
    harmful_train = random.sample(load_dataset_split(harmtype='harmful', split='train', instructions_only=True), cfg.n_train)
    harmless_train = random.sample(load_dataset_split(harmtype='harmless', split='train', instructions_only=True), cfg.n_train)
    harmful_val = random.sample(load_dataset_split(harmtype='harmful', split='val', instructions_only=True), cfg.n_val)
    harmless_val = random.sample(load_dataset_split(harmtype='harmless', split='val', instructions_only=True), cfg.n_val)
    print(harmful_val[0])
    print(harmless_val[0])
    return harmful_train, harmless_train, harmful_val, harmless_val

def filter_data(cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val):
    """
    Filter datasets based on refusal scores.

    Returns:
        Filtered datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """
    def filter_examples(dataset, scores, threshold, comparison):
        return [inst for inst, score in zip(dataset, scores.tolist()) if comparison(score, threshold)]

    print('filtering data')
    print(f'  num harmless_train before: {len(harmless_train)}')
    print(f'  num harmless_val before: {len(harmless_val)}')
    print(f'  num harmful_train before: {len(harmful_train)}')
    print(f'  num harmful_val before: {len(harmful_val)}')
    if cfg.filter_train:
        harmful_train_scores = get_refusal_scores(model_base.model, harmful_train, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmless_train_scores = get_refusal_scores(model_base.model, harmless_train, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmful_train = filter_examples(harmful_train, harmful_train_scores, 0, lambda x, y: x > y)
        harmless_train = filter_examples(harmless_train, harmless_train_scores, 0, lambda x, y: x < y)

    if cfg.filter_val:
        harmful_val_scores = get_refusal_scores(model_base.model, harmful_val, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmless_val_scores = get_refusal_scores(model_base.model, harmless_val, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmful_val = filter_examples(harmful_val, harmful_val_scores, 0, lambda x, y: x > y)
        harmless_val = filter_examples(harmless_val, harmless_val_scores, 0, lambda x, y: x < y)
    
    print(f'  num harmless_train after: {len(harmless_train)}')
    print(f'  num harmless_val after: {len(harmless_val)}')
    print(f'  num harmful_train after: {len(harmful_train)}')
    print(f'  num harmful_val after: {len(harmful_val)}')

    return harmful_train, harmless_train, harmful_val, harmless_val

def generate_and_save_candidate_directions(cfg, model_base, harmful_train, harmless_train):
    """Generate and save candidate directions."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'generate_directions')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'generate_directions'))

    mean_diffs = generate_directions(
        model_base,
        harmful_train,
        harmless_train,
        artifact_dir=os.path.join(cfg.artifact_path(), "generate_directions"))

    torch.save(mean_diffs, os.path.join(cfg.artifact_path(), 'generate_directions/mean_diffs.pt'))

    return mean_diffs

def select_and_save_direction(cfg, model_base, harmful_val, harmless_val, candidate_directions):
    """Select and save the direction."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'select_direction')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'select_direction'))

    pos, layer, direction = select_direction(
        model_base,
        harmful_val,
        harmless_val,
        candidate_directions,
        artifact_dir=os.path.join(cfg.artifact_path(), "select_direction")
    )

    with open(f'{cfg.artifact_path()}/direction_metadata_layer{layer}_pos{pos}.json', "w") as f:
        json.dump({"pos": pos, "layer": layer}, f, indent=4)

    torch.save(direction, f'{cfg.artifact_path()}/direction_layer{layer}_pos{pos}.pt')

    return pos, layer, direction

def generate_and_save_completions_for_dataset(cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label, dataset_name, dataset=None, suffix=None):
    """Generate and save completions for a dataset."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), f'completions{suffix}')):
        os.makedirs(os.path.join(cfg.artifact_path(), f'completions{suffix}'))

    if dataset is None:
        dataset = load_dataset(dataset_name)

    completions = model_base.generate_completions(dataset, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, max_new_tokens=cfg.max_new_tokens)
    with open(f'{cfg.artifact_path()}/completions{suffix}/{dataset_name}_{intervention_label}_completions.json', "w") as f:
        json.dump(completions, f, indent=4)

def evaluate_completions_and_save_results_for_dataset(cfg, intervention_label, dataset_name, eval_methodologies, suffix=None):
    """Evaluate completions and save results for a dataset."""
    with open(os.path.join(cfg.artifact_path(), f'completions{suffix}/{dataset_name}_{intervention_label}_completions.json'), 'r') as f:
        completions = json.load(f)

    evaluation = evaluate_jailbreak(
        completions=completions,
        methodologies=eval_methodologies,
        evaluation_path=os.path.join(cfg.artifact_path(), f"completions{suffix}", f"{dataset_name}_{intervention_label}_evaluations.json"),
    )

    with open(f'{cfg.artifact_path()}/completions{suffix}/{dataset_name}_{intervention_label}_evaluations.json', "w") as f:
        json.dump(evaluation, f, indent=4)

def evaluate_loss_for_datasets(cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label, suffix):
    """Evaluate loss on datasets."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'loss_evals')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'loss_evals'))

    on_distribution_completions_file_path = os.path.join(cfg.artifact_path(), f'completions{suffix}/harmless_baseline_completions.json')

    loss_evals = evaluate_loss(model_base, fwd_pre_hooks, fwd_hooks, batch_size=cfg.ce_loss_batch_size, n_batches=cfg.ce_loss_n_batches, completions_file_path=on_distribution_completions_file_path)

    with open(f'{cfg.artifact_path()}/loss_evals/{intervention_label}_loss_eval.json', "w") as f:
        json.dump(loss_evals, f, indent=4)

def run_pipeline(model_path, vector_path, completion_suffix, no_eval, no_generate):
    """Run the full pipeline."""
    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias, model_path=model_path)

    # Load and sample datasets
    harmful_train, harmless_train, harmful_val, harmless_val = load_and_sample_datasets(cfg)

    model_base = construct_model_base(cfg.model_path)
    
    # Filter datasets based on refusal scores
    harmful_train, harmless_train, harmful_val, harmless_val = filter_data(cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val)

    if vector_path is not None:
        print(f'vector path: {vector_path}')
        data = torch.load(vector_path)
        if isinstance(data, dict):
            # should be sparse vector
            pos = None
            layer = data['layer']
            direction = data['direction']
            if completion_suffix is None:
                completion_suffix = "_" + os.path.splitext(os.path.basename(vector_path))[0]
        
        elif type(data) == torch.Tensor:
            # should be learned vector
            direction = data
            # get layer and norm
            try:
                metadata_path = os.path.splitext(vector_path)[0] + "_metadata.json"
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    layer = metadata['layer']
                val_vector_path = os.path.splitext(vector_path)[0] + "_val.json"
                with open(val_vector_path, 'r') as f:
                    val_data = json.load(f)
                    dim_norm = val_data['norm']
                direction = direction / direction.norm() * dim_norm

            except Exception as e:
                print(f'smth went wrong: {e}')
                print('using default dim layer')
                layer = {
                    'google/gemma-2-2b-it': 15,
                    'meta-llama/Llama-3.2-3B-Instruct': 12,
                    'Qwen/Qwen3-8B': 20
                }[model_path]
        
            if completion_suffix is None:
                completion_suffix = "_" + os.path.basename(os.path.dirname(vector_path))
    else:
        # 1. Generate candidate refusal directions
        candidate_directions = generate_and_save_candidate_directions(cfg, model_base, harmful_train, harmless_train)
        # 2. Select the most effective refusal direction
        pos, layer, direction = select_and_save_direction(cfg, model_base, harmful_val, harmless_val, candidate_directions)
        if completion_suffix is None:
            completion_suffix = f"_layer{layer}_pos{pos}"
        print(f'direction: {direction[:100]}')

    baseline_fwd_pre_hooks, baseline_fwd_hooks = [], []
    ablation_fwd_pre_hooks, ablation_fwd_hooks = get_all_direction_ablation_hooks(model_base, direction)
    actadd_fwd_pre_hooks, actadd_fwd_hooks = [(model_base.model_block_modules[layer], get_activation_addition_input_pre_hook(vector=direction, coeff=-1.0))], []

    # 3a. Generate and save completions on harmful evaluation datasets
    if not no_generate:
        for dataset_name in cfg.evaluation_datasets:
            # generate_and_save_completions_for_dataset(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline', dataset_name, suffix=completion_suffix)
            # generate_and_save_completions_for_dataset(cfg, model_base, ablation_fwd_pre_hooks, ablation_fwd_hooks, 'ablation', dataset_name, suffix=completion_suffix)
            generate_and_save_completions_for_dataset(cfg, model_base, actadd_fwd_pre_hooks, actadd_fwd_hooks, 'actadd', dataset_name, suffix=completion_suffix)

    # 3b. Evaluate completions and save results on harmful evaluation datasets
    if not no_eval:
        for dataset_name in cfg.evaluation_datasets:
            # evaluate_completions_and_save_results_for_dataset(cfg, 'baseline', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies, suffix=completion_suffix)
            # evaluate_completions_and_save_results_for_dataset(cfg, 'ablation', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies, suffix=completion_suffix)
            evaluate_completions_and_save_results_for_dataset(cfg, 'actadd', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies, suffix=completion_suffix)
    
    # 4a. Generate and save completions on harmless evaluation dataset
    harmless_test = random.sample(load_dataset_split(harmtype='harmless', split='test'), cfg.n_test)

    if not no_generate:
        # generate_and_save_completions_for_dataset(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline', 'harmless', dataset=harmless_test, suffix=completion_suffix)
        
        actadd_refusal_pre_hooks, actadd_refusal_hooks = [(model_base.model_block_modules[layer], get_activation_addition_input_pre_hook(vector=direction, coeff=+1.0))], []
        generate_and_save_completions_for_dataset(cfg, model_base, actadd_refusal_pre_hooks, actadd_refusal_hooks, 'actadd', 'harmless', dataset=harmless_test, suffix=completion_suffix)

    if not no_eval:
        # 4b. Evaluate completions and save results on harmless evaluation dataset
        # evaluate_completions_and_save_results_for_dataset(cfg, 'baseline', 'harmless', eval_methodologies=cfg.refusal_eval_methodologies, suffix=completion_suffix)
        evaluate_completions_and_save_results_for_dataset(cfg, 'actadd', 'harmless', eval_methodologies=cfg.refusal_eval_methodologies, suffix=completion_suffix)

    # 5. Evaluate loss on harmless datasets
    evaluate_loss_for_datasets(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline', completion_suffix)
    # evaluate_loss_for_datasets(cfg, model_base, ablation_fwd_pre_hooks, ablation_fwd_hooks, 'ablation', completion_suffix)
    evaluate_loss_for_datasets(cfg, model_base, actadd_fwd_pre_hooks, actadd_fwd_hooks, 'actadd', completion_suffix)

if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(model_path=args.model_path, 
                 vector_path=args.vector_path, 
                 completion_suffix=args.completion_suffix, 
                 no_eval=args.no_eval,
                 no_generate = args.no_generate
    )
