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
      directories: List of directory paths to search
      pattern: File pattern to match (default: '*.pt')
    
    Returns:
      List of matching file paths
    """
    files = []
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
