import torch


def enable_dropout(model):
    """Enable dropout layers during inference time.

    Args:
        model (nn.Module): Pytorch model.
    """
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


def lde(log_a, log_b):
    max_log = torch.max(log_a, log_b)
    min_log = torch.min(log_a, log_b)
    return max_log + torch.log(1 + torch.exp(min_log - max_log))
