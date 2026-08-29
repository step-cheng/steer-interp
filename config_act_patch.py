from dataclasses import dataclass, fields, asdict
import os
import argparse
import time

dim_layer_pos_dict = {
    'google/gemma-2-2b-it': {15: (15, -1), 14: (14, -2), 10: (10, -5)},
    'google/gemma-2-9b-it': {23: (23, -1), 22: (22, -1)},
    'meta-llama/Llama-3.2-3B-Instruct': {12: (12, -4)},
    'Qwen/Qwen3-8B': {20: (20, -8)}
}
dim_layer_map = {
    'google/gemma-2-2b-it': 15,
    'google/gemma-2-9b-it': 23,
    'meta-llama/Llama-3.2-3B-Instruct': 12,
    'Qwen/Qwen3-8B': 20
}

@dataclass
class Config:
    exp_name: str
    model_path: str = "google/gemma-2-2b-it"
    steer_flag: bool = False
    harm_flag: bool = False
    n_steps: int = 10
    trapezoidal: bool = True
    method: str = "ig"
    use_ln_grad: bool = False
    concept: str = "refusal"
    metric: str = 'dirKL'
    kl_min_thresh: float = 0
    kl_max_thresh: float = float('inf')
    aggregation: str = "sum"
    num_examples: int = 100
    check_negs: bool = False
    nodes_only: bool = False
    save_grads: bool = False
    layer: int = 15
    learn_type: str = 'dim'
    learn_path: str = None

    def update_params_from_parser(self, args):
        """Update config parameters from argparse namespace."""
        for field in fields(self):
            if hasattr(args, field.name):
                value = getattr(args, field.name)
                if value is not None:  # Only update if explicitly provided
                    setattr(self, field.name, value)
            elif hasattr(args, f"not_{field.name}"):
                value = getattr(args, f"not_{field.name}")
                assert type(value) == bool
                if value is not None:
                    setattr(self, field.name, not value)
            else:
                raise ValueError(f"misalignment in config attribute {field.name} and parser params")

        self.save_dir = f"/fs/nexus-scratch/scheng03/steer-interp-results/circuits/{self.model_path.split('/')[-1]}/{self.exp_name}"
        self.layer, self.pos = dim_layer_pos_dict[self.model_path][self.layer]

        return self
    
    def update_save_dir(self):
        if os.path.exists(self.save_dir):
            current_time = time.strftime("%m-%d-%Y-%H:%M:%S", time.gmtime())
            self.save_dir += current_time
    
    @property
    def concept_params(self):
        concept_params = {
            'refusal': {'harm_flag': self.harm_flag}
        }[self.concept]
        return concept_params
    
    @property
    def kl_range(self):
        return (self.kl_min_thresh, self.kl_max_thresh)

    @property
    def concept_suffix(self):
        if self.learn_type == 'dim':
            suffix = {
                "refusal": f"{('harmful' if self.harm_flag else 'harmless')}q_{('steered' if self.steer_flag else 'base')}" +
                            f"_layer{self.layer}_pos{self.pos}"
            }[self.concept]
        else:
            suffix = {
                'refusal': f"{('harmful' if self.harm_flag else 'harmless')}q_{('steered' if self.steer_flag else 'base')}" +
                            f"_{self.learn_path}"
            }[self.concept]
        return suffix

    @property
    def kl_data_path(self):
        data_path = f"data/kl_divs/{self.concept}/{self.model_path.split('/')[-1]}"
        f_name = f"kl_analysis_{self.concept_suffix}.parquet"
        return os.path.join(data_path, f_name)
    
    def to_dict(self, include_properties: bool = True):
        d = asdict(self)
        if self.kl_max_thresh == float('inf'):
            d['kl_max_thresh'] = None
        if include_properties:
            d['save_dir'] = self.save_dir
            d['kl_data_path'] = self.kl_data_path
            d['concept_suffix'] = self.concept_suffix
            d['kl_range'] = self.kl_range
            if self.kl_max_thresh == float('inf'): 
                d['kl_range'] = (d['kl_range'][0], None)
        return d
    
    @classmethod
    def from_dict(cls, d: dict):
        """Create Config from a dictionary (e.g., loaded from JSON)."""
        # Handle inf values stored as strings
        if d.get('kl_max_thresh') is None:
            d['kl_max_thresh'] = float('inf')
        
        # Filter to only valid fields (in case dict has extra keys like properties)
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        
        return cls(**filtered)

    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--n_steps', type=int, default=10)
    parser.add_argument('--layer', type=int, default=15)
    parser.add_argument('--not_trapezoidal', action="store_true", default=False)
    parser.add_argument('--steer_flag', action='store_true', default=False)
    parser.add_argument('--harm_flag', action='store_true', default=False)
    parser.add_argument('--method', type=str, default='ig', choices=['ig', 'exact', 'ig2'])
    parser.add_argument('--use_ln_grad', action='store_true', default=False)
    parser.add_argument('--concept', type=str, default="refusal")
    parser.add_argument('--check_negs', action="store_true", default=False)
    parser.add_argument('--metric', type=str, default='logit', choices=['dirKL', 'nodirKL', 'logit'])
    parser.add_argument('--kl_min_thresh', type=int, default=0)
    parser.add_argument('--kl_max_thresh', type=int, default=float('inf'))
    parser.add_argument('--model_path', type=str, default='google/gemma-2-2b-it')
    parser.add_argument('--aggregation', type=str, default='sum')
    parser.add_argument('--num_examples', type=int, default=100)
    parser.add_argument('--nodes_only', action="store_true", default=False)
    parser.add_argument('--save_grads', action="store_true", default=False)
    parser.add_argument('--learn_type', type=str, default='dim', choices=['dim', 'ntp', 'reps', 'ortho'])
    parser.add_argument('--learn_path', type=str, default=None)
    args = parser.parse_args()

    return Config(args.exp_name).update_params_from_parser(args)