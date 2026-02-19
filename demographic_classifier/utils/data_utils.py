from typing import List, Dict
import numpy as np
from sklearn.metrics import roc_auc_score

def dict_avg(list_of_dicts: List[Dict]) -> Dict:

    first_dict = list_of_dicts[0]

    result_dict = {}

    for key, value in first_dict.items():

        if isinstance(value, list):
            # leaf node copute element-avg
            avg_list = []
            
            for idx in range(len(value)):
                avg_list.append(np.mean([d[key][idx] for d in list_of_dicts]))

            result_dict[key] = avg_list

        elif isinstance(value, dict):
            # recursive call
            result_dict[key] = dict_avg([d[key] for d in list_of_dicts])

    return result_dict


def permutation_test(targets, predictions, n=1_000):
    targets, predictions = np.array(targets), np.array(predictions)
    auc_obs = roc_auc_score(y_true=targets, y_score=predictions)

    # Ensuring that 1 permutation = actual predictions
    permulation_auc_scores = [auc_obs]

    # permuting predictions and calculating AUC
    for _ in range(n - 1):
        np.random.shuffle(predictions)
        auc_perm = roc_auc_score(y_true=targets, y_score=predictions)
        permulation_auc_scores.append(auc_perm)

    # Summarizing test results
    permulation_auc_scores = np.array(permulation_auc_scores)
    auc_baseline = permulation_auc_scores.mean()
    p_value = (permulation_auc_scores >= auc_obs).sum() / n

    return auc_obs, auc_baseline, p_value

