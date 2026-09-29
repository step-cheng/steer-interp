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
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--freeze_type', type=str, default=None)
    return parser.parse_args()


def evaluate_freeze_completions_and_save_results_for_dataset(cfg, eval_methodologies, path):
    """Evaluate completions and save results for a dataset."""
    try:
        with open(path, 'r') as f:
            completions = json.load(f)
    except:
        with open(path, 'r') as f:
            content = f.read()

        # The file is concatenated pretty-printed JSON objects
        decoder = json.JSONDecoder()
        completions = []
        idx = 0
        while idx < len(content):
            content = content[idx:].lstrip()
            if not content:
                break
            obj, end = decoder.raw_decode(content)
            completions.append(obj)
            idx = end

        for completion in completions:
            if 'freeze' in completion and 'response' not in completion:
                completion['response'] = completion['freeze']
                del completion['freeze']

    evaluation = evaluate_jailbreak(
        completions=completions,
        methodologies=eval_methodologies,
        evaluation_path=path,
    )
    splits = path.split('.')
    base_path = '.'.join(splits[:-1])
    save_path = f"{base_path}_evaluations.json"
    with open(save_path, "w") as f:
        json.dump(evaluation, f, indent=4)
        print(f'saved to {save_path}')


def run_pipeline(model_path, dataset, freeze_type):
    """Run the full pipeline."""
    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias, model_path=model_path)

    harm_flag = {
        'jailbreakbench': True,
        'strongreject': True,
        'alpaca': False
    }[dataset]
    eval_methodologies = cfg.jailbreak_eval_methodologies if harm_flag else cfg.refusal_eval_methodologies

    model_name = model_path.split('/')[-1]
    path = f"freeze_generations/{model_name}/{freeze_type}_{dataset}.jsonl"
    evaluate_freeze_completions_and_save_results_for_dataset(cfg, eval_methodologies=eval_methodologies, path=path)

if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(
        model_path=args.model_path, 
        dataset=args.dataset, 
        freeze_type=args.freeze_type
    )
