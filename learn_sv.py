"""
train the model on harmless and harmful generations at the same time
"""
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from utils.gen_utils import set_seeds
from utils.training_utils import _handle_cross_entropy_loss, _handle_reps_loss
from tqdm import tqdm
from config_act_patch import dim_layer_pos_dict
import os
import random
from utils.data_utils import load_steering_vector
import json
from dataclasses import dataclass, asdict
import argparse
from pipeline.submodules.select_direction import refusal_score

VECTOR_DIR = "trained_vectors/{model_name}"
REFUSAL_TOKS = {
    'meta-llama/Llama-3.2-3B-Instruct': [40],
    'google/gemma-2-2b-it': [235285],
    'Qwen/Qwen3-8B': [40],
}


@dataclass
class TrainConfig():
    model_path: str = None
    exp_name: str = None
    epochs: int = 10
    lr: float = 0.01
    intervention_layer: int = None
    steering_factors: list = None
    batch_size: int = 6
    mini_batch_size: int = 6
    grad_clip: float = 1.0
    l1_decay: float = 0
    l2_decay: float = 0
    scale_harmful: float = 1
    scale_harmless: float = 1
    dtype = t.float32
    simpo_scaler: float = 2e-2
    loss_type: str = 'ntp'
    save_interval: int = 1
    num_tokens: int = 64
    val_batch_size: int = 2
    concept: str = "refusal"
    seed: int = 42
    ref_vector_paths: list = None

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--concept', type=str, default=self.concept)
        parser.add_argument('--model_path', type=str, required=True)
        parser.add_argument('--exp_name', type=str, required=True)
        parser.add_argument('--epochs', type=int, default=self.epochs)
        parser.add_argument('--lr', type=float, default=self.lr)
        parser.add_argument('--intervention_layer', type=int, default=self.intervention_layer)
        parser.add_argument('--steering_factors', type=float, nargs='+', default=self.steering_factors)
        parser.add_argument('--batch_size', type=int, default=self.batch_size)
        parser.add_argument('--mini_batch_size', type=int, default=self.mini_batch_size)
        parser.add_argument('--val_batch_size', type=int, default=self.val_batch_size)
        parser.add_argument('--grad_clip', type=float, default=self.grad_clip)
        parser.add_argument('--l1_decay', type=float, default=self.l1_decay)
        parser.add_argument('--l2_decay', type=float, default=self.l2_decay)
        parser.add_argument('--scale_harmful', type=float, default=self.scale_harmful)
        parser.add_argument('--scale_harmless', type=float, default=self.scale_harmless)
        parser.add_argument('--dtype', type=str, default="float32", choices=['float16', 'bfloat16', 'float32'])
        parser.add_argument('--simpo_scaler', type=float, default=self.simpo_scaler)
        parser.add_argument('--loss_type', type=str, default=self.loss_type, choices=['ntp', 'reps', 'ortho'])
        parser.add_argument('--ref_vector_paths', type=str, nargs='*', default=self.ref_vector_paths)
        parser.add_argument('--save_interval', type=int, default=self.save_interval)
        parser.add_argument('--num_tokens', type=int, default=self.num_tokens)
        parser.add_argument('--seed', type=int, default=self.seed)
        parser.add_argument('--load_pretrained', action='store_true', default=False)
        parser.add_argument('--pbar', action='store_true', default=False)
        args = parser.parse_args()

        dtype_map = {'float16': t.float16, 'bfloat16': t.bfloat16, 'float32': t.float32}
        for k, v in vars(args).items():
            if k == 'dtype':
                setattr(self, k, dtype_map[v])
            else:
                setattr(self, k, v)
        
        # Validate accumulation settings
        assert self.batch_size % self.mini_batch_size == 0, \
            f"batch_size ({self.batch_size}) must be divisible by mini_batch_size ({self.mini_batch_size})"

        return self, args.load_pretrained, args.pbar
    


class RefusalDataset(Dataset):
    """
    Dataset for refusal steering vector training.
    
    completions_harmful are compliant completions on harmful questions.
    completions_harmless are refused completions on harmless questions
    """
    def __init__(self, all_datasets_w, all_datasets_l, dataset_coeffs):
    # def __init__(self, completions_harmful_comply, completions_harmful_refuse, completions_harmless_comply, completions_harmless_refuse):
        self.data = []
        assert len(all_datasets_w) == len(all_datasets_l) == len(dataset_coeffs)
        num_prompt_sets = len(all_datasets_w)
        for i in range(num_prompt_sets):
            dataset_w, dataset_l = all_datasets_w[i], all_datasets_l[i]
            assert len(dataset_w) == len(dataset_l)

            for completion_w, completion_l in zip(dataset_w, dataset_l):
                assert completion_w['prompt'] == completion_l['prompt']
                winning = [{"role": "user", "content": completion_w['prompt']},
                        {"role": "assistant", "content": completion_w['response']}]
                losing = [{"role": "user", "content": completion_l['prompt']},
                        {"role": "assistant", "content": completion_l['response']}] 
                self.data.append({
                    'winning': winning,
                    'losing': losing,
                    'coeff': dataset_coeffs[i],
                    'beta': None,
                }) 

    def set_betas(self, betas):
        assert len(betas) == len(self.data)
        for i, beta in enumerate(betas):
            self.data[i]['beta'] = beta

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class SteeringVectorTrainer:
    def __init__(
        self, 
        concept: str,
        model_path: str,
        train_dataset: RefusalDataset,
        val_dataset: RefusalDataset,
        config: TrainConfig,
        load_pretrained: bool,
        use_pbar: bool
    ):
        # save path
        save_dir = VECTOR_DIR.format(model_name=model_path.split('/')[-1])
        self.save_path = os.path.join(save_dir, config.exp_name)
        if not load_pretrained:
            if os.path.exists(os.path.join(self.save_path)):
                print(f"exp dir {config.exp_name} already exists, making new one")
                import time
                self.save_path += time.strftime("%M-%d-%Y_%H-%M-%S", time.localtime())
            os.makedirs(self.save_path, exist_ok=True)
        else:
            assert os.path.exists(self.save_path)
            with open(os.path.join(self.save_path, "config.json"), 'r') as f:
                config_as_dict = json.load(f)
                config = TrainConfig(**config_as_dict)
        self.config = config
        self.device = t.device('cuda' if t.cuda.is_available() else 'cpu')
        self.dtype = config.dtype
        self.disable_pbar = not use_pbar
        
        # Load model (frozen)
        self.model_path = model_path
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map='auto', 
            torch_dtype=config.dtype,
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        if not self.model.is_gradient_checkpointing:
            print('setting gradient checkpointing')
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache=False
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        
        # Config
        self.hidden_size = self.model.config.hidden_size
        self.intervention_layer = config.intervention_layer
        self.steering_factors = config.steering_factors or [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        self.epochs = config.epochs
        self.grad_clip = config.grad_clip
        
        # Initialize steering vector (small random init)
        self.vector = nn.Parameter(
            t.randn(self.hidden_size, device=self.device, dtype=config.dtype) * 0.01
        )
        self.load_dim_norm()
        
        self.optimizer = t.optim.AdamW([self.vector], lr=config.lr, weight_decay=0)
        self.scheduler = t.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters = config.epochs
        )
        self.l1_decay = config.l1_decay
        self.l2_decay = config.l2_decay
        self.scale_harmful = config.scale_harmful
        self.scale_harmless = config.scale_harmless
        self.loss_type = config.loss_type
        self.ref_vectors = self.load_ref_vectors(config.ref_vector_paths) if self.loss_type == 'ortho' else None
        
        # Gradient accumulation
        self.mini_batch_size = config.mini_batch_size
        self.batch_size = config.batch_size
        self.val_batch_size = config.val_batch_size
        self.accum_steps = config.batch_size // config.mini_batch_size

        # Dataloader — uses mini_batch_size for actual forward passes
        self.num_tokens = config.num_tokens

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        # Hook state
        self._steering_vec_to_add = None
        self._hook_handle = None

        self.save_interval = config.save_interval

    def _collate_fn(self, batch):
        """Collate batch into padded tensors."""
        messages_w_list = []
        messages_l_list = []
        coeffs = []
        betas = []

        for sample in batch:
            messages_w_list.append(sample['winning'])
            messages_l_list.append(sample['losing'])
            coeffs.append(sample['coeff'])
            betas.append(sample.get('beta', None))

        inputs_w, prompt_lens_w = self._tokenize_batch(messages_w_list)
        inputs_l, prompt_lens_l = self._tokenize_batch(messages_l_list)

        return {
            'inputs_w': inputs_w,
            'inputs_l': inputs_l,
            'prompt_lens': prompt_lens_w,  # same prompt, so shared
            'coeffs': t.tensor(coeffs, dtype=self.dtype),
            'betas': t.tensor(betas, dtype=self.dtype) if betas[0] is not None else None,
        }
    
    @property
    def vector_norm(self):
        return self.vector.norm().item()
    
    def _get_target_layer(self):
        """Get module for intervention. Adjust for your architecture."""
        # Gemma-2: model.model.layers[i]
        # Llama: model.model.layers[i]
        return self.model.model.layers[self.intervention_layer]
    
    def _steering_hook(self, module, inputs):
        """Add steering vector to residual stream output."""
        
        # outputs is (hidden_states, ...) for most models
        if isinstance(inputs, tuple):
            hidden_states = inputs[0].detach() # cut backward graph here
            # Broadcast: [batch, hidden] -> [batch, 1, hidden] + [batch, seq, hidden]
            modified = hidden_states + self._steering_vec_to_add.unsqueeze(1)
            return (modified,) + inputs[1:]
        else:
            return inputs.detach() + self._steering_vec_to_add.unsqueeze(1)
    
    def _register_hook(self):
        layer = self._get_target_layer()
        self._hook_handle = layer.register_forward_pre_hook(self._steering_hook)
    
    def _remove_hook(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def _tokenize_batch(self, messages_list):
        """
        Tokenize a list of conversations with left-padding for batch processing.

        Returns:
            input_ids: [batch_size, max_seq_len]
            attention_mask: [batch_size, max_seq_len]
            prompt_lens: list[int]
        """
        prompt_lens = []

        for messages in messages_list:
            prompt = [messages[0]]
            prompt_text = self.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True)
            prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)['input_ids']
            prompt_lens.append(len(prompt_ids))

        full_text_list = self.tokenizer.apply_chat_template(
            messages_list, tokenize=False, add_generation_prompt=False)
        full_inputs = self.tokenizer(full_text_list, return_tensors='pt', padding=True, add_special_tokens=False)
        
        return full_inputs, t.tensor(prompt_lens)

    def _compute_ref_logps_from_logits(self, logits, input_ids, prompt_len, pad_offset):
        """Compute masked sum of log-probs from pre-computed logits for a single sample."""
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        seq_len = shift_labels.shape[1]

        response_start = max(0, pad_offset + prompt_len - 1)
        end = min(response_start + self.num_tokens, seq_len)

        # Slice to response tokens only — avoids full-sequence log_softmax
        log_probs = t.gather(
            shift_logits[0, response_start:end, :].log_softmax(-1),
            dim=1,
            index=shift_labels[0, response_start:end].unsqueeze(1)
        ).squeeze(1)

        return log_probs.sum()

    def _precompute_betas(self, dataset, simpo_scaler=1.0):
        """
        Precompute RePS scaling factors using the frozen reference model.
        
        For each (winning, losing) pair:
            beta = max((ref_logp_losing - ref_logp_winning) * simpo_scaler, 1.0)
        
        Must be called without the steering hook active.
        """
        data_loader = DataLoader(
            dataset, 
            batch_size=self.val_batch_size, 
            shuffle=False,
            collate_fn=self._collate_fn,
        )
        new_betas = []
        beta_records = []
        with t.no_grad():
            for batch in tqdm(data_loader, desc="Precomputing betas", disable=self.disable_pbar):
                inputs_w = {k: v.to(self.device) for k, v in batch['inputs_w'].items()}
                inputs_l = {k: v.to(self.device) for k, v in batch['inputs_l'].items()}
                prompt_lens = batch['prompt_lens']
                coeffs = batch['coeffs']

                logits_w = self.model(**inputs_w).logits
                logits_l = self.model(**inputs_l).logits

                # Per-sample log-probs (loop is cheap, forward pass already batched)
                for i in range(logits_w.shape[0]):
                    attn_w = inputs_w['attention_mask'][i]
                    attn_l = inputs_l['attention_mask'][i]

                    # Skip leading pad tokens
                    pad_offset_w = (attn_w == 0).sum().item()
                    pad_offset_l = (attn_l == 0).sum().item()

                    prompt_len = prompt_lens[i].item()
                    ref_logps_w = self._compute_ref_logps_from_logits(
                        logits_w[i:i+1], inputs_w['input_ids'][i:i+1], prompt_len, pad_offset_w)
                    ref_logps_l = self._compute_ref_logps_from_logits(
                        logits_l[i:i+1], inputs_l['input_ids'][i:i+1], prompt_len, pad_offset_l)

                    raw_diff = (ref_logps_l - ref_logps_w).item()
                    beta = max(raw_diff * simpo_scaler, 1.0)
                    new_betas.append(beta)

                    # Decode prompt and responses without special tokens
                    ids_w = inputs_w['input_ids'][i]
                    ids_l = inputs_l['input_ids'][i]
                    prompt_text = self.tokenizer.decode(
                        ids_w[pad_offset_w : pad_offset_w + prompt_len], skip_special_tokens=True)
                    response_w = self.tokenizer.decode(
                        ids_w[pad_offset_w + prompt_len:], skip_special_tokens=True)
                    response_l = self.tokenizer.decode(
                        ids_l[pad_offset_l + prompt_len:], skip_special_tokens=True)

                    coeff = coeffs[i].item()
                    beta_records.append({
                        'index': len(beta_records),
                        'type': 'harmful' if coeff < 0 else 'harmless',
                        'coeff': coeff,
                        'prompt': prompt_text.strip(),
                        'winning_response': response_w.strip(),
                        'losing_response': response_l.strip(),
                        'ref_logps_winning': ref_logps_w.item(),
                        'ref_logps_losing': ref_logps_l.item(),
                        'raw_diff': raw_diff,
                        'simpo_scalar': simpo_scaler,
                        'beta': beta,
                    })
        
        # Save diagnostic JSON
        beta_dump_path = os.path.join(self.save_path, "beta_diagnostics.json")
        with open(beta_dump_path, 'w') as f:
            json.dump(beta_records, f, indent=2, ensure_ascii=False)
        print(f"Saved beta diagnostics to {beta_dump_path}")

        print(f"Precomputed {len(new_betas)} betas | "
            f"mean={sum(new_betas)/len(new_betas):.3f} | "
            f"min={min(new_betas):.3f} | max={max(new_betas):.3f}")

        return new_betas
    
    def _handle_ortho_proj(self):
        with t.no_grad():
            for ref_vector in self.ref_vectors:
                ref_hat = ref_vector / ref_vector.norm()
                proj = (self.vector @ ref_hat) * ref_hat
                self.vector.data -= proj

    def train(self):
        # DataLoader uses mini_batch_size for memory-efficient forward passes
        data_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.mini_batch_size, 
            shuffle=True,
            collate_fn=self._collate_fn,
        )
        
        if self.loss_type == 'reps':
            self._remove_hook()
            betas = self._precompute_betas(dataset=self.train_dataset, simpo_scaler=getattr(self.config, 'simpo_scaler', 1.0))
            self.train_dataset.set_betas(betas)

        self._register_hook()
        
        try:
            epoch_losses = []
            epoch_losses_reg = []
            epoch_losses_harmful = []
            epoch_losses_harmless = []
            epoch_lrs = []
            for epoch in range(self.epochs):
                total_loss = 0.0
                total_loss_reg = 0.0
                total_loss_obj_harmful = 0.0
                total_loss_obj_harmless = 0.0
                num_samples = 0
                num_harmful_samples = 0
                num_harmless_samples = 0
                current_lr = self.optimizer.param_groups[0]['lr']
                
                pbar = tqdm(data_loader, desc=f'Epoch {epoch+1}/{self.epochs}', disable=self.disable_pbar)
                
                self.optimizer.zero_grad()
                mini_batch_idx = 0
                
                for batch in pbar:
                    bsz = batch['coeffs'].shape[0]

                    # Process each sample (batch_size=1 recommended for simplicity)
                    factors = t.tensor(
                        [random.choice(self.steering_factors) for _ in range(bsz)],
                        device=self.device, dtype=self.dtype
                    )
                    
                    # Set steering vector for hook
                    coeffs = batch['coeffs'].to(self.device)
                    # [bsz] * [bsz] -> [bsz] -> [bsz, 1] * [hidden] -> [bsz, hidden]
                    self._steering_vec_to_add = (coeffs * factors).unsqueeze(1) * self.vector
                    
                    inputs_w = batch['inputs_w'].to(self.device)
                    
                    # Forward pass
                    if self.loss_type in ['ntp', 'ortho']:
                        outputs = self.model(**inputs_w)
                        loss_objective = t.stack([
                            _handle_cross_entropy_loss(
                                outputs.logits[i:i+1], inputs_w['input_ids'][i:i+1],
                                batch['prompt_lens'][i], self.num_tokens,
                                pad_offset=(inputs_w['attention_mask'][i] == 0).sum().item())
                            for i in range(bsz)
                        ])

                    elif self.loss_type == 'reps':
                        inputs_l = batch['inputs_l'].to(self.device)
                        outputs_w = self.model(**inputs_w)
                        outputs_l = self.model(**inputs_l)

                        loss_objective = t.stack([_handle_reps_loss(
                            outputs_w.logits[i:i+1], outputs_l.logits[i:i+1],
                            batch['betas'][i], inputs_w['input_ids'][i:i+1], inputs_l['input_ids'][i:i+1], 
                            batch['prompt_lens'][i], self.num_tokens,
                            pad_offset_w=(inputs_w['attention_mask'][i] == 0).sum().item(),
                            pad_offset_l=(inputs_l['attention_mask'][i] == 0).sum().item())
                            for i in range(bsz)
                        ])

                    # Regularization (applied once per accumulation step, scaled accordingly)
                    loss_reg = t.tensor(0.0, device=self.device)
                    if self.l1_decay > 0:
                        loss_reg = loss_reg + self.l1_decay * self.vector.abs().sum()
                    if self.l2_decay > 0:
                        loss_reg = loss_reg + self.l2_decay * self.vector.pow(2).sum()
                    
                    # Backward
                    scales = t.where(
                        coeffs > 0,
                        t.tensor(self.scale_harmless, device=self.device, dtype=self.dtype),
                        t.tensor(self.scale_harmful, device=self.device, dtype=self.dtype),
                    )
                    scaled_loss_objective = loss_objective * scales

                    # Scale loss by accumulation steps so gradient magnitude is
                    # equivalent to a single large-batch forward pass
                    loss = (scaled_loss_objective.mean() + loss_reg) / self.accum_steps
                    loss.backward()

                    mini_batch_idx += 1

                    # --- Optimizer step after accumulating enough mini-batches ---
                    if mini_batch_idx % self.accum_steps == 0:
                        t.nn.utils.clip_grad_norm_([self.vector], self.grad_clip)
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                        if self.loss_type == 'ortho':
                            self._handle_ortho_proj()

                    # --- Logging (unscaled for interpretability) ---
                    with t.no_grad():
                        harmful_mask = coeffs < 0
                        harmless_mask = coeffs > 0
                        total_loss += (scaled_loss_objective.sum() + loss_reg*bsz).item()
                        total_loss_reg += loss_reg.item() * bsz
                        if harmful_mask.any():
                            total_loss_obj_harmful += scaled_loss_objective[harmful_mask].sum().item()
                            num_harmful_samples += harmful_mask.sum().item()
                        if harmless_mask.any():
                            total_loss_obj_harmless += scaled_loss_objective[harmless_mask].sum().item()
                            num_harmless_samples += harmless_mask.sum().item()
                        num_samples += bsz
                        
                        pbar.set_postfix({
                            'loss': f'{loss.item():.4f}',
                            'vec_norm': f'{self.vector_norm:.2f}',
                            'lr': f'{current_lr:.4f}'
                        })

                # Flush any remaining accumulated gradients at epoch end
                if mini_batch_idx % self.accum_steps != 0:
                    t.nn.utils.clip_grad_norm_([self.vector], self.grad_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.loss_type == 'ortho':
                        self._handle_ortho_proj()

                self.scheduler.step()
                
                avg_loss = total_loss / max(num_samples, 1)
                avg_loss_reg = total_loss_reg / max(num_samples, 1)
                avg_loss_harmful = total_loss_obj_harmful / max(num_harmful_samples, 1)
                avg_loss_harmless = total_loss_obj_harmless / max(num_harmless_samples, 1)
                epoch_losses.append(avg_loss)
                epoch_losses_harmful.append(avg_loss_harmful)
                epoch_losses_harmless.append(avg_loss_harmless)
                epoch_losses_reg.append(avg_loss_reg)
                epoch_lrs.append(current_lr)

                tqdm.write(f'Epoch {epoch+1}: avg_loss={avg_loss:.4f}, vec_norm={self.vector_norm:.4f}, lr={current_lr:.4f}')


                if epoch % self.save_interval == 0:
                    self.save(epoch, avg_loss)
        
        finally:
            self._remove_hook()
            # --- Plotting ---
            import matplotlib.pyplot as plt
            import seaborn as sns

            sns.set_theme(style="ticks", context="notebook", palette="deep")
            fig, axs = plt.subplots(1, 2, figsize=(8, 4.5))

            epochs_range = list(range(1, len(epoch_losses) + 1))
            axs[0].plot(epochs_range, epoch_losses, linewidth=2, color=sns.color_palette()[0], label='total')
            axs[0].plot(epochs_range, epoch_losses_harmless, linewidth=2, color=sns.color_palette()[1], label=f'harmless {self.scale_harmless}')
            axs[0].plot(epochs_range, epoch_losses_harmful, linewidth=2, color=sns.color_palette()[2], label=f'harmful {self.scale_harmful}')
            axs[0].plot(epochs_range, epoch_losses_reg, linewidth=2, color=sns.color_palette()[3], label=f'reg L1 {self.l1_decay}, L2 {self.l2_decay}')

            axs[0].set_xlabel("Epoch", fontsize=12)
            axs[0].set_ylabel("Average Loss", fontsize=12)
            axs[0].set_title("Steering Vector Training Loss", fontsize=14, fontweight='bold')
            axs[0].legend(frameon=False)

            axs[1].plot(epochs_range, epoch_lrs, linewidth=2)
            axs[1].set_yscale('log')
            axs[1].set_xlabel("Epoch", fontsize=12)
            axs[1].set_ylabel("Learning Rate", fontsize=12)
            axs[1].set_title("Steering Vector Training Loss", fontsize=14, fontweight='bold')

            for ax in axs.flat:
                sns.despine(ax=ax, top=True, right=True)

            plt.tight_layout()
            plt.savefig(os.path.join(self.save_path, "training_loss.png"), dpi=150, bbox_inches='tight')
            plt.show()
        
        return self.vector.detach().clone()
    
    def eval(self, epochs:int, eval_all_steer_factors:bool=False):
        print(f"steer vec norm: {self.vector_norm}")
        val_data_loader = DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            collate_fn=self._collate_fn
        )
        
        self._register_hook()

        def _eval_steer_factor(steer_factor):
            print(f'testing steer factor {steer_factor}')
            total_loss = 0.0
            total_loss_reg = 0.0
            total_loss_obj_harmful = 0.0
            total_loss_obj_harmless = 0.0
            num_samples = 0
            num_harmful_samples = 0
            num_harmless_samples = 0

            match_scores = []
            all_coeffs = []

            with t.no_grad():
                pbar = tqdm(val_data_loader, disable=self.disable_pbar)
                for batch in pbar:
                    coeffs = batch['coeffs'].to(self.device)
                    inputs_w = batch['inputs_w'].to(self.device)
                    bsz = coeffs.shape[0]

                    # normalize vector to diff means, or sweep across steering factors
                    self._steering_vec_to_add = (coeffs * steer_factor).unsqueeze(1) * self.vector

                    # Forward pass
                    if self.loss_type in ['ntp', 'ortho']:
                        outputs_w = self.model(**inputs_w)
                        loss_objective = t.stack([
                            _handle_cross_entropy_loss(
                                outputs_w.logits[i:i+1], inputs_w['input_ids'][i:i+1],
                                batch['prompt_lens'][i], self.num_tokens,
                                pad_offset=(inputs_w['attention_mask'][i] == 0).sum().item())
                            for i in range(bsz)
                        ])

                    elif self.loss_type == 'reps':
                        inputs_l = batch['inputs_l'].to(self.device)
                        outputs_w = self.model(**inputs_w)
                        outputs_l = self.model(**inputs_l)

                        loss_objective = t.stack([_handle_reps_loss(
                            outputs_w.logits[i:i+1], outputs_l.logits[i:i+1],
                            batch['betas'][i], inputs_w['input_ids'][i:i+1], inputs_l['input_ids'][i:i+1], 
                            batch['prompt_lens'][i], self.num_tokens,
                            pad_offset_w=(inputs_w['attention_mask'][i] == 0).sum().item(),
                            pad_offset_l=(inputs_l['attention_mask'][i] == 0).sum().item())
                            for i in range(bsz)
                        ])

                    for i in range(bsz):
                        pad_offset = (inputs_w['attention_mask'][i] == 0).sum().item()
                        # Position of logits that predict the first completion token
                        first_logit_pos = pad_offset + batch['prompt_lens'][i] - 1

                        gen_logits = outputs_w.logits[i:i+1, first_logit_pos:-1]
                        pred_toks = t.argmax(gen_logits, dim=-1)
                        y_toks = inputs_w['input_ids'][i:i+1, first_logit_pos+1:]

                        match_mask = pred_toks.flatten() == y_toks.flatten()
                        match_score = (match_mask.sum()/match_mask.numel()).item()
                        if coeffs[i] > 0: # positive steer case
                            match_scores.append(match_score)
                        else:
                            match_scores.append(match_score)
                        all_coeffs.append(coeffs[i].item())

                    # Regularization (same as training)
                    loss_reg = t.tensor(0.0, device=self.device)
                    if self.l1_decay > 0:
                        loss_reg = loss_reg + self.l1_decay * self.vector.abs().sum()
                    if self.l2_decay > 0:
                        loss_reg = loss_reg + self.l2_decay * self.vector.pow(2).sum()
                    
                    # Scale losses (same as training)
                    scales = t.where(
                        coeffs > 0,
                        t.tensor(self.scale_harmless, device=self.device, dtype=self.dtype),
                        t.tensor(self.scale_harmful, device=self.device, dtype=self.dtype),
                    )
                    scaled_loss_objective = loss_objective * scales

                    # Accumulate statistics
                    harmful_mask = coeffs < 0
                    harmless_mask = coeffs > 0
                    total_loss += (scaled_loss_objective.sum() + loss_reg * bsz).item()
                    total_loss_reg += loss_reg.item() * bsz
                    if harmful_mask.any():
                        total_loss_obj_harmful += scaled_loss_objective[harmful_mask].sum().item()
                        num_harmful_samples += harmful_mask.sum().item()
                    if harmless_mask.any():
                        total_loss_obj_harmless += scaled_loss_objective[harmless_mask].sum().item()
                        num_harmless_samples += harmless_mask.sum().item()
                    num_samples += bsz
                    
                    pbar.set_postfix({'loss': f'{scaled_loss_objective.mean().item():.4f}'})
            
            # Compute averages
            avg_loss = total_loss / max(num_samples, 1)
            avg_loss_reg = total_loss_reg / max(num_samples, 1)
            avg_loss_harmful = total_loss_obj_harmful / max(num_harmful_samples, 1)
            avg_loss_harmless = total_loss_obj_harmless / max(num_harmless_samples, 1)
            avg_match_score = t.mean(t.tensor(match_scores, dtype=t.float32)).item()

            print(f'Avg Loss: {avg_loss}, Avg Match Score: {avg_match_score}')

            return avg_loss, avg_loss_reg, avg_loss_harmful, avg_loss_harmless, avg_match_score

        if not eval_all_steer_factors:
            # eval on dim mean steer vec norm
            dim_steer_factor = self.diffmeans_norm / self.vector_norm
            avg_loss, _, _, _, avg_match_score = _eval_steer_factor(dim_steer_factor)
            results = {
                'norm': self.diffmeans_norm,
                'loss': avg_loss if isinstance(avg_loss, float) else avg_loss.item(),
                'match_score': avg_match_score if isinstance(avg_match_score, float) else avg_match_score.item(),
            }

            with open(os.path.join(self.save_path, f"direction_{epochs}_val.json"), 'w') as f:
                json.dump(results, f, indent=4)
            
            self._remove_hook()
            return avg_loss, avg_match_score, self.diffmeans_norm
        else:
            avg_losses = t.zeros(len(self.steering_factors)+1, dtype=t.float32)
            avg_match_scores = t.zeros(len(self.steering_factors)+1, dtype=t.float32)
            # eval across train steer factors
            for i, steer_factor in enumerate(self.steering_factors):
                avg_loss, avg_loss_reg, avg_loss_harmful, avg_loss_harmless, avg_match_score = _eval_steer_factor(steer_factor)
                avg_losses[i] = avg_loss
                avg_match_scores[i] = avg_match_score

            # eval on dim mean steer vec norm
            dim_steer_factor = self.diffmeans_norm / self.vector_norm
            avg_loss, avg_loss_reg, avg_loss_harmful, avg_loss_harmless, avg_match_score = _eval_steer_factor(dim_steer_factor)
            avg_losses[-1] = avg_loss
            avg_match_scores[-1] = avg_match_score

            all_norms = [s*self.vector_norm for s in self.steering_factors] + [dim_steer_factor*self.vector_norm]
            best_avg_loss_ind = t.argmin(avg_losses)
            best_match_score_ind = t.argmax(avg_match_scores)
            best_loss_norm = all_norms[best_avg_loss_ind]
            best_match_norm = all_norms[best_match_score_ind]

            # get ranks
            loss_ranks = t.argsort(t.argsort(avg_losses))
            match_ranks = t.argsort(t.argsort(-avg_match_scores))
            combined_ranks = loss_ranks + match_ranks
            best_combined_ind = t.argmin(combined_ranks)
            best_norm = all_norms[best_combined_ind]
            print(f"best norms: overall {best_norm}, by match {best_match_norm}, by loss {best_loss_norm}")

            results = {
                'best_norm': best_norm,
                'avg_loss': avg_losses.tolist(),
                'avg_match_score': avg_match_scores.tolist(),
                'all_steering_factors': self.steering_factors + [dim_steer_factor],
                'all_norms': all_norms,
            }

            with open(os.path.join(self.save_path, f"direction_{epochs}_val.json"), 'w') as f:
                json.dump(results, f, indent=4)

            self._remove_hook()

            return avg_losses[best_combined_ind], avg_match_scores[best_combined_ind], best_norm

    def eval_all_epochs(self):
        all_epoch_losses = t.zeros(self.epochs, dtype=t.float32)
        all_epoch_matches = t.zeros(self.epochs, dtype=t.float32)
        all_epoch_norms = t.zeros(self.epochs, dtype=t.float32)

        # Handle reps loss beta precomputation for val set
        if self.loss_type == 'reps':
            self._remove_hook()
            betas = self._precompute_betas(
                simpo_scaler=getattr(self.config, 'simpo_scaler', 1.0),
                dataset=self.val_dataset
            )
            self.val_dataset.set_betas(betas)

        for e in range(self.epochs):
            print(f"evaluating epoch {e}")
            self.load_vector(e)
            epoch_val_loss, epoch_val_match, epoch_norm = self.eval(e)
            all_epoch_losses[e] = epoch_val_loss
            all_epoch_matches[e] = epoch_val_match
            all_epoch_norms[e] = epoch_norm

        # rank best differently based off loss type
        if self.loss_type in ['ntp', 'ortho']:
            loss_ranks = t.argsort(t.argsort(all_epoch_losses))
            match_ranks = t.argsort(t.argsort(-all_epoch_matches))
            combined_ranks = loss_ranks + match_ranks
            best_epoch = t.argmin(combined_ranks)
        elif self.loss_type == 'reps':
            loss_ranks = t.argsort(t.argsort(all_epoch_losses))
            combined_ranks = loss_ranks
            best_epoch = t.argmin(combined_ranks)
        best_norm = all_epoch_norms[best_epoch]

        val_results = {
            'best_norm': best_norm.item(),
            'best_epoch': best_epoch.item(),
            'best_loss': all_epoch_losses[best_epoch].item(),
            'best_match': all_epoch_matches[best_epoch].item(),
        }

        with open(os.path.join(self.save_path, f"val.json"), 'w') as f:
            json.dump(val_results, f)

    def load_vector(self, epoch=None):
        if not os.path.exists(self.save_path):
            raise ValueError(f"specified save path does not exist {self.save_path}")
        if epoch is None: epoch = config.epochs-1
        print(f'loading vector from epoch {epoch}')
        
        vector_info = t.load(os.path.join(self.save_path, f"direction_{epoch}.pt"))
        if type(vector_info) == dict:
            vector = vector_info['vector']
        elif type(vector_info) == t.Tensor:
            vector = vector_info
        
        with open(os.path.join(self.save_path, f"direction_{epoch}_metadata.json"), 'r') as f:
            metadata = json.load(f)


        self.vector = nn.Parameter(vector.to(self.device).to(self.dtype))
        self.intervention_layer = metadata['layer']

    def load_ref_vectors(self, ref_vector_paths):
        if not ref_vector_paths:
            return []
        return [t.load(p, map_location=self.device).to(self.dtype) for p in ref_vector_paths]

    def load_dim_norm(self):
        try:
            layer, pos = dim_layer_pos_dict[self.model_path][self.intervention_layer]
            diffmeans_vector, _, _ = load_steering_vector("refusal", self.model_path, {'harm_flag': True}, pos, layer)
            self.diffmeans_norm = diffmeans_vector.norm().item()
            print(f'Diff Mean vector norm: {self.diffmeans_norm}')
        except:
            print('Could not load Diff Means Vector, setting to None')
            self.diffmeans_norm = None
    
    def save(self, epochs=None, avg_loss=None):
        """Save learned vector and config."""
        t.save(
            self.vector.detach().cpu()
        , os.path.join(self.save_path, f"direction_{epochs}.pt"))

        with open(os.path.join(self.save_path, f"direction_{epochs}_metadata.json"), 'w') as f:
            metadata = {
                'layer': self.intervention_layer,
                'norm': self.vector_norm,
                'avg_loss': avg_loss
            }
            json.dump(metadata, f, indent=4)
        
        with open(os.path.join(self.save_path, "config.json"), 'w') as f:
            json.dump(asdict(self.config), f, indent=4)

        print(f'Saved to {self.save_path}')

    def check_generations(self, test_norm:float, data:str="val"):
        if data == 'train':
            dataset = self.train_dataset
        elif data == 'val':
            dataset = self.val_dataset
        else:
            print('unknown data type, using val data')
            dataset = self.val_dataset

        self._register_hook()
        for sample in tqdm(dataset):
            # Set steering vector for hook
            prompt = [sample['winning'][0]]
            prompt_template = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            prompt_inputs = self.tokenizer(prompt_template, return_tensors='pt', add_special_tokens=False)
            coeff = sample['coeff']
            self._steering_vec_to_add = (coeff * test_norm / self.vector_norm) * self.vector.unsqueeze(0)
            
            outputs = self.model.generate(**prompt_inputs.to(self.device), max_new_tokens=200)
            
            [print(self.tokenizer.decode(o)) for o in outputs]
        self._remove_hook()

if __name__ == '__main__':
    
    config, load_pretrained, pbar = TrainConfig().parse_args()
    set_seeds(config.seed)

    save = True

    # Load your data
    path = "training_data/{model_name}/{harm_type}_{response_type}_{data_type}.json"
    model_name = config.model_path.split('/')[-1]
    with open(path.format(model_name=model_name, harm_type="harmful", response_type="comply", data_type="train"), 'r') as f:
        train_harmful_comply = json.load(f)
    with open(path.format(model_name=model_name, harm_type="harmful", response_type="refuse", data_type="train"), 'r') as f:
        train_harmful_refuse = json.load(f)
    with open(path.format(model_name=model_name, harm_type="harmless", response_type="comply", data_type="train"), 'r') as f:
        train_harmless_comply = json.load(f)
    with open(path.format(model_name=model_name, harm_type="harmless", response_type="refuse", data_type="train"), 'r') as f:
        train_harmless_refuse = json.load(f)
    with open(path.format(model_name=model_name, harm_type="harmful", response_type="comply", data_type="val"), 'r') as f:
        val_harmful_comply = json.load(f)
    with open(path.format(model_name=model_name, harm_type="harmful", response_type="refuse", data_type="val"), 'r') as f:
        val_harmful_refuse = json.load(f)
    with open(path.format(model_name=model_name, harm_type="harmless", response_type="refuse", data_type="val"), 'r') as f:
        val_harmless_refuse = json.load(f)
    with open(path.format(model_name=model_name, harm_type="harmless", response_type="comply", data_type="val"), 'r') as f:
        val_harmless_comply = json.load(f)

    val_dataset = RefusalDataset(
        all_datasets_w=[val_harmful_comply, val_harmless_refuse],
        all_datasets_l=[val_harmful_refuse, val_harmless_comply],
        dataset_coeffs = [-1, 1],
    )
    train_dataset = RefusalDataset(
        all_datasets_w = [train_harmful_comply, train_harmless_refuse],
        all_datasets_l = [train_harmful_refuse, train_harmless_comply],
        dataset_coeffs = [-1, 1],
    )

    trainer = SteeringVectorTrainer(
        concept=config.concept,
        model_path=config.model_path,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        load_pretrained=load_pretrained,
        use_pbar=pbar
    )


    if load_pretrained:
        print("loading trained steering vector")
        trainer.load_vector()
    else:
        learned_vector = trainer.train()

    
    trainer.eval_all_epochs()
    # trainer.check_generations(116)
    
    

