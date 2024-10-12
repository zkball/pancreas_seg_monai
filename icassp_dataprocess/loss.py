import torch
import torch.nn as nn
import torch.nn.functional as F

def _take_channels(*xs, ignore_channels=None):
    if ignore_channels is None:
        return xs
    else:
        channels = [channel for channel in range(xs[0].shape[1]) if channel not in ignore_channels]
        xs = [torch.index_select(x, dim=1, index=torch.tensor(channels).to(x.device)) for x in xs]
        return xs


def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def f_score(pr, gt, beta=1, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate F-score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        beta (float): positive constant
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: F score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp

    score = ((1 + beta**2) * tp + eps) / ((1 + beta**2) * tp + beta**2 * fn + fp + eps)

    return score

class DiceLoss(nn.Module):
    def __init__(self, eps=1.0, beta=1.0, ignore_channels=None):
        super(DiceLoss, self).__init__()
        self.eps = eps
        self.beta = beta
        self.ignore_channels = ignore_channels

    def forward(self, out, gt):
        return 1 - f_score(
            out,
            gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )



class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2., reduction='none'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        target_tensor = target_tensor.squeeze(1).long()
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )





class RankedWeightedLoss(nn.Module):
    def __init__(self, base_loss_fn, max_weight=2.0):
        super(RankedWeightedLoss, self).__init__()
        self.base_loss_fn = base_loss_fn
        self.max_weight = max_weight

    def forward(self, preds, masks, rank_ratios):
        """
        Compute the weighted loss with weights based on ranks.

        Args:
        - preds (torch.Tensor): Predictions of shape (B, C, H, W).
        - masks (torch.Tensor): Ground truth masks of shape (B, H, W).
        - ranks (torch.Tensor): Ranks of the slices of shape (B,).

        Returns:
        - loss (torch.Tensor): Weighted loss value.
        """
        # weights = self.max_weight / (ranks + 1).float()  # Higher rank (smaller index) gets higher weight
        weights = 1 - rank_ratios.float()

        loss = 0.0
        for i in range(preds.shape[0]):
            weight = weights[i]
            if not torch.any(masks[i]):
                weight = 0.5   # learn no tumor contents
                # print("fixed loss")
            loss += weight * self.base_loss_fn(preds[i], masks[i])

        return loss / preds.shape[0]