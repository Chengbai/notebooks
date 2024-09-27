import torch


from common_util import get_logger

logger = get_logger(__name__)
# img/text constrastive loss


def constrastive_logit_loss(contrastive_logit: torch.tensor) -> torch.tensor:
    """
    inputs:
        - contrastive_logit: N x IMG_TOKENS(TXT_TOKENS)
    output:
        - loss: torch.tensor
    """
    e = 1e-6
    device = contrastive_logit.device
    N, IMG_TOKENS = contrastive_logit.size()

    # for exp stability
    max_v = contrastive_logit.max()
    contrastive_logit_table = contrastive_logit - max_v

    logit_exp = contrastive_logit_table.exp()

    # Note: Following mask logic could put the loss negative!
    # Comment out for more thinking.
    # mask = torch.eye(N, dtype=torch.bool, device=device)
    # masked_logit_exp = logit_exp.where(~mask, 0.0).to(device)

    masked_logit_exp = logit_exp
    masked_logit_exp_agg = masked_logit_exp.sum(dim=-1, keepdim=True)
    target = torch.arange(N, device=device)
    target_exp_logits = logit_exp[torch.arange(N).unsqueeze(1), target.unsqueeze(1)]
    loss = -((target_exp_logits / (masked_logit_exp_agg + e)).sum() / N).log()
    if torch.isnan(loss):
        logger.info(
            f"nan loss: \ntarget_exp_logits: {target_exp_logits}\ncontrastive_logit: {contrastive_logit}\n contrastive_logit_table: {contrastive_logit_table}\n logit_exp: {logit_exp}"
        )
    return loss
