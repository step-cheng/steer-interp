import torch as t
import torch.nn.functional as F
from tqdm import tqdm

def _handle_cross_entropy_loss(logits, input_ids, prompt_len, num_loss_tokens, pad_offset=0):
    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    
    # Mask: only compute loss after prompt
    bsz, seq_len = shift_labels.shape
    start = max(pad_offset + prompt_len - 1, 0)  # -1 due to shift
    end = min(start + num_loss_tokens, seq_len)
    
    # Only compute log-probs for the target tokens — avoids [bsz*seq, vocab] materialization
    loss = F.cross_entropy(
        shift_logits[0, start:end, :],
        shift_labels[0, start:end],
    )
    
    return loss

def _handle_reps_loss(logits_steer_w, logits_steer_l, beta, input_ids_w, input_ids_l, prompt_len, num_loss_tokens, pad_offset_w=0, pad_offset_l=0):
    # Shift for next-token prediction
    shift_logits_w = logits_steer_w[:, :-1, :]
    shift_logits_l = logits_steer_l[:, :-1, :]
    shift_labels_w = input_ids_w[:, 1:]
    shift_labels_l = input_ids_l[:, 1:]

    # Masks: only compute over response tokens (may differ in length)
    response_start_w = max(0, pad_offset_w + prompt_len - 1)  # -1 due to shift
    response_start_l = max(0, pad_offset_l + prompt_len - 1)  # -1 due to shift

    seq_len_w = shift_labels_w.shape[1]
    mask_w = t.zeros(seq_len_w, device=shift_labels_w.device, dtype=shift_labels_w.dtype)
    num_w = min(num_loss_tokens, seq_len_w - response_start_w)
    mask_w[response_start_w:response_start_w + num_w] = 1.0

    seq_len_l = shift_labels_l.shape[1]
    mask_l = t.zeros(seq_len_l, device=shift_labels_l.device, dtype=shift_labels_l.dtype)
    num_l = min(num_loss_tokens, seq_len_l - response_start_l)
    mask_l[response_start_l:response_start_l + num_l] = 1.0

    # Per-token log probs (each against its own labels)
    log_probs_w = t.gather(
        shift_logits_w.log_softmax(-1), dim=2, index=shift_labels_w.unsqueeze(2)
    ).squeeze(2)
    log_probs_l = t.gather(
        shift_logits_l.log_softmax(-1), dim=2, index=shift_labels_l.unsqueeze(2)
    ).squeeze(2)

    # Masked sum of log probs
    chosen_logps = (log_probs_w * mask_w.unsqueeze(0)).sum(dim=-1)
    rejected_logps = (log_probs_l * mask_l.unsqueeze(0)).sum(dim=-1)

    # Length-normalized, beta-scaled SimPO
    n_w = mask_w.sum().clamp(min=1)
    n_l = mask_l.sum().clamp(min=1)

    scaled_chosen = (beta / n_w) * chosen_logps
    scaled_rejected = (1.0 / n_l) * rejected_logps

    loss = -F.logsigmoid(scaled_chosen - scaled_rejected)
    return loss.mean()

def _handle_bireps_loss(logits_steer_w, logits_steer_l, logits_null_w, logits_null_l,
                      beta_pos, beta_neg,
                      input_ids_w, input_ids_l,
                      prompt_len, num_loss_tokens,
                      pad_offset_w=0, pad_offset_l=0):
    """
    Full RePS loss (equation 8): L = -E[log σ(Δ⁺) + log σ(Δ⁻)]

    Positive direction (eq. 5): Φ_Steer applied, winning=steered (w), losing=original (l)
    Negative direction (eq. 6): Φ_Null applied, winning=original (l), losing=steered (w)

    Args:
        logits_steer_w: logits from Φ_Steer forward pass on steered response y^c
        logits_steer_l: logits from Φ_Steer forward pass on original response y
        logits_null_w:  logits from Φ_Null forward pass on original response y
        logits_null_l:  logits from Φ_Null forward pass on steered response y^c
        beta_pos: max(log p(y|x) - log p(y^c|x), 1)
        beta_neg: max(log p(y^c|x) - log p(y|x), 1)
    """
    # Δ⁺: Φ_Steer, winning = y^c, losing = y
    loss_pos = _handle_reps_loss(
        logits_w=logits_steer_w, logits_l=logits_steer_l,
        beta=beta_pos,
        input_ids_w=input_ids_w, input_ids_l=input_ids_l,
        prompt_len=prompt_len, num_loss_tokens=num_loss_tokens,
        pad_offset_w=pad_offset_w, pad_offset_l=pad_offset_l,
    )

    # Δ⁻: Φ_Null, winning = y (original), losing = y^c (steered)
    # Note the swap: w and l are flipped relative to positive direction
    loss_neg = _handle_reps_loss(
        logits_w=logits_null_l, logits_l=logits_null_w,
        beta=beta_neg,
        input_ids_w=input_ids_l, input_ids_l=input_ids_w,  # swapped
        prompt_len=prompt_len, num_loss_tokens=num_loss_tokens,
        pad_offset_w=pad_offset_l, pad_offset_l=pad_offset_w,  # swapped
    )

    return loss_pos + loss_neg