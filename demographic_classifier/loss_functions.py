from typing import Dict
import torch
import torch.nn as nn
from torch.nn import BCELoss, BCEWithLogitsLoss, MSELoss, Module, CrossEntropyLoss


class AdaptiveBalancedBCELoss(Module):
    def __init__(self, eps=1e-6):
        """
        Adaptive Balanced BCE Loss: Automatically computes positive & negative class weights in each batch.
        :param eps: Small value to avoid division by zero.
        """
        super(AdaptiveBalancedBCELoss, self).__init__()
        self.eps = eps

    def forward(self, logits, targets):
        """
        Compute the adaptive balanced BCE loss.
        :param logits: Predicted logits (before sigmoid)
        :param targets: Ground truth labels (0 or 1)
        :return: Weighted BCE loss
        """
        # Compute the number of positive and negative samples
        pos_count = targets.sum() + self.eps
        neg_count = (targets.shape[0] - targets.sum()) + self.eps

        # Compute the adaptive weights
        pos_weight = neg_count / (pos_count + neg_count)  # Balances positive samples
        neg_weight = pos_count / (pos_count + neg_count)  # Balances negative samples

        # Compute standard BCE loss
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )

        # Apply dynamic weights
        weights = targets * pos_weight + (1 - targets) * neg_weight
        balanced_loss = weights * loss

        return balanced_loss.mean()
    
    
class FocalLoss(Module):
    """
    For y=1: L = -(1-p)^α * log(p)
    For y=0: L = -(1-y)^β * p^α * log(1-p)
    
    where:
    - p is the predicted probability
    - y is the ground truth label
    - α and β are focusing parameters that down-weight easy examples
    """
    def __init__(self, alpha=2, beta=4, eps=1e-6, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.reduction = reduction
        
        assert alpha > 0, 'alpha must be greater than 0'
        assert beta > 0, 'beta must be greater than 0'
        assert reduction in ['mean', 'sum'], 'reduction must be one of "mean", "sum"'
        
    def forward(self, predictions, targets):
        """
        Compute the focal loss.
        :param predictions: Predicted probabilirt
        :param targets: Ground truth labels (between 0 and 1)
        :return: Focal loss
        """
        # masks of positive and negative samples
        positive_mask = targets == 1
        negative_mask = ~positive_mask
        
        loss_pos = torch.log(predictions[positive_mask] + self.eps) * (1 - predictions[positive_mask]) ** self.alpha
        loss_neg = torch.log(1 - predictions[negative_mask] + self.eps) * predictions[negative_mask] ** self.alpha * (1 - predictions[negative_mask]) ** self.beta
        
        loss = -1 * torch.cat([loss_pos, loss_neg], dim=0)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

loss_function_MAP = {
    'BCELoss': BCELoss,
    'BCEWithLogitsLoss': BCEWithLogitsLoss,
    'AdaptiveBalancedBCELoss': AdaptiveBalancedBCELoss,
    'MSELoss': MSELoss,
    'CrossEntropyLoss': CrossEntropyLoss,
    'FocalLoss': FocalLoss,
}

def setup_loss_function(loss_function: str, loss_function_init_args: Dict) -> Module:

    if loss_function not in loss_function_MAP:
        raise ValueError(f"{loss_function} loss not founds. Select from {list(loss_function_MAP.keys())}")
    
    return loss_function_MAP[loss_function](**loss_function_init_args)
