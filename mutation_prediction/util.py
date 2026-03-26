import torch
import numpy as np
import torch.nn as nn
from sklearn import metrics
from copy import deepcopy
from typing import Literal
import glob
import os

def roc_auc_score_nan(y_true, y_scores):
  """
  Calculates the ROC AUC score, returning NaN if only one class is present.
  """
  try:
    return metrics.roc_auc_score(y_true, y_scores)
  except ValueError:
    return np.nan

def list_files(directories, pattern='*.pt'):
    """
    This function lists all files matching a pattern in given directories.
    
    Args:
      directories: String path or list of directory paths to search
      pattern: File pattern to match (default: '*.pt')
    
    Returns:
      List of matching file paths
    """
    files = []
    # Handle both string and list inputs
    if isinstance(directories, str):
        directories = [directories]
    
    for directory in directories:
        files.extend(glob.glob(os.path.join(directory, pattern)))
    return files

def loss_function(loss_func, preds, targets, group):
    loss = 0.
    group_of_loss = {}
    group_length = {}
    loss_func_noavg = deepcopy(loss_func)
    loss_func_noavg.reduction = 'none'
    loss = loss_func_noavg(preds, targets)
    # get unique groups
    unique_groups = list(set(group))
    for g in unique_groups:
        idx = [i for i, x in enumerate(group) if x == g]
        group_of_loss[g] = loss[idx].mean()
    loss = loss.mean()
    return loss, group_of_loss

def case_insensitive_glob(pattern):
    """
    This function performs a case-insensitive glob.
    
    Args:
      pattern: The glob pattern to match.
    
    Returns:
      A list of matching filenames.
    """
    def either(c):
        return '[%s%s]' % (c.lower(), c.upper()) if c.isalpha() else c
    return glob.glob(''.join(map(either, pattern)))


def weibull_loss(shape, scale, time, event):
    """
    Weibull distribution loss for survival analysis.
    """
    y = time
    u = event
    a = scale
    b = shape
    hazard0 = (y + 1e-35/a)**b
    hazard1 = (y + 1/a)**b
    return -torch.mean(u * torch.log(torch.exp(hazard1 - hazard0) - 1) - hazard1)


class GroupStratifiedSurvivalLoss(nn.Module):
    """
    Group-stratified loss for survival analysis tasks.
    Computes both overall loss and per-group losses using Weibull distribution.
    """
    def __init__(self):
        super(GroupStratifiedSurvivalLoss, self).__init__()

    def forward(self, shape, scale, time, event, lengths, group):
        loss = 0.
        group_of_loss = {}
        group_length = {}
        lengths = list(lengths)
        for i in range(len(lengths)):
            loss += weibull_loss(shape[i], scale[i], time[i], event[i])
            if group[i] not in group_of_loss:
                group_length[group[i]] = 1
                group_of_loss[group[i]] = weibull_loss(
                    shape[i], scale[i], time[i], event[i])
            else:
                group_length[group[i]] += 1
                group_of_loss[group[i]
                              ] += weibull_loss(shape[i], scale[i], time[i], event[i])
        loss = loss / len(lengths)
        group_of_loss = {k: v / group_length[k]
                         for k, v in group_of_loss.items()}
        return loss, group_of_loss


class GroupStratifiedLoss(nn.Module):
    """
    Group-stratified loss for classification tasks.
    Computes both overall loss and per-group losses.
    """
    def __init__(self, loss_fn):
        super(GroupStratifiedLoss, self).__init__()
        self.loss_func_noavg = deepcopy(loss_fn)
        self.loss_func_noavg.reduction = 'none'

    def forward(self, preds, targets, group):
        loss = 0.
        group_of_loss = {}
        group_length = {}
        loss = self.loss_func_noavg(preds, targets)

        unique_groups = list(set(group))
        for g in unique_groups:
            idx = [i for i, x in enumerate(group) if x == g]
            group_of_loss[g] = loss[idx].mean()
        loss = loss.mean()
        return loss, group_of_loss
