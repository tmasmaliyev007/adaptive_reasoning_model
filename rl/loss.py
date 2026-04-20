import torch
import torch.nn.functional as F

def perplexity_loss(logits, labels, num_items_in_batch=None):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    vocab_size = shift_logits.size(-1)
    flat_logits = shift_logits.view(-1, vocab_size)
    flat_labels = shift_labels.view(-1)

    ce_per_token = F.cross_entropy(
        flat_logits,
        flat_labels,
        ignore_index=-100,
        reduction="mean"
    )

    ppl = torch.exp(ce_per_token)
    return ppl