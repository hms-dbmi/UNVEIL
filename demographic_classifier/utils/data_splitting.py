from sklearn.model_selection import train_test_split, KFold
import json
import numpy as np
import re

def _group_id(key: str, pattern: re.Pattern):
    m = pattern.match(key)
    if not m:
        raise ValueError(f"Key '{key}' did not match split_regex.")
    return m.group(0)

def train_val_test_random_split(sample_keys, validation_size, test_size, split_regex: str = r".+", random_state=None):
    """
    Groups keys by split_regex (group 0), then splits by groups.
    """
    pattern = re.compile(split_regex)

    # Extract unique group ids
    group_ids = [_group_id(k, pattern) for k in sample_keys]
    unique_group_ids = list(set(group_ids))

    remaining_size = validation_size + test_size
    if remaining_size <= 0 or remaining_size >= 1:
        raise ValueError("validation_size + test_size must be in (0, 1).")

    # Split into train and (val+test)
    train_groups, val_test_groups = train_test_split(
        unique_group_ids, test_size=remaining_size, random_state=random_state
    )

    # Split (val+test) into val and test
    test_size_adjusted = test_size / remaining_size  # proportion of TEST inside the remaining chunk
    val_groups, test_groups = train_test_split(
        val_test_groups, test_size=test_size_adjusted, random_state=random_state
    )

    # Map back to keys
    train_sample_keys = [k for k in sample_keys if _group_id(k, pattern) in train_groups]
    val_sample_keys   = [k for k in sample_keys if _group_id(k, pattern) in val_groups]
    test_sample_keys  = [k for k in sample_keys if _group_id(k, pattern) in test_groups]

    return train_sample_keys, val_sample_keys, test_sample_keys


def train_val_test_cv_split(sample_keys, val_from_train_ratio, n_splits=5, random_state=None, split_regex: str = r".+"):
    """
    Yields folds using KFold over group ids (defined by split_regex group 0).
    Each fold: split train_val groups further into train/val by val_from_train_ratio.
    """
    if not (0 < val_from_train_ratio < 1):
        raise ValueError("val_from_train_ratio must be in (0, 1).")

    pattern = re.compile(split_regex)

    # Unique group ids
    unique_group_ids = np.array(list(set(_group_id(k, pattern) for k in sample_keys)))

    # KFold over group ids
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for train_val_index, test_index in kf.split(unique_group_ids):
        train_val_groups = unique_group_ids[train_val_index]
        test_groups = unique_group_ids[test_index]

        # Split train vs val within the training portion
        train_groups, val_groups = train_test_split(
            train_val_groups, test_size=val_from_train_ratio, random_state=random_state
        )

        train_sample_keys = [k for k in sample_keys if _group_id(k, pattern) in set(train_groups)]
        val_sample_keys   = [k for k in sample_keys if _group_id(k, pattern) in set(val_groups)]
        test_sample_keys  = [k for k in sample_keys if _group_id(k, pattern) in set(test_groups)]

        yield train_sample_keys, val_sample_keys, test_sample_keys


def train_val_test_from_file_split(sample_keys, split_file_path):

    with open(split_file_path, "r") as f:
        split = json.load(f)

    train_samples, val_samples, test_samples = split['train'], split['val'], split['test']

    # Splitting the original sample_keys into train, val, and test sets based on unique sample IDs
    train_sample_keys = [k for k in sample_keys if k in train_samples]
    val_sample_keys = [k for k in sample_keys if k in val_samples]
    test_sample_keys = [k for k in sample_keys if k in test_samples]

    return train_sample_keys, val_sample_keys, test_sample_keys
