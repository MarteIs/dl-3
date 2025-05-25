import torch

def mrr(target: torch.Tensor, predicted: torch.Tensor) -> float:
    target_expanded = target.unsqueeze(1).expand_as(predicted)

    correct_mask = (target_expanded == predicted)
    recipropal_ranks = torch.zeros_like(target, dtype=torch.float)

    for i in range(correct_mask.size(0)):
        match = torch.where(correct_mask[i])[0]

        if len(match) > 0:
            recipropal_ranks[i] = 1.0 / (match[0].item() + 1)

    return recipropal_ranks.mean().item()
