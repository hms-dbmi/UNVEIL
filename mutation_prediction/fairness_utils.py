import pandas as pd
from sklearn import metrics
import numpy as np
# import albumentations as albu
# import cv2
import random
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report
from  typing import Literal
# argument library
import argparse
import sklearn
import os

random.seed(24)

HIGHER_BETTER_COLS=['AUC', 'ACC', 'TPR', 'TNR', 'PPV', 'NPV', 'PQD', 'PQD(class)','PR','NR','BAcc',
                    'EPPV', 'ENPV', 'DPM(Positive)', 'DPM(Negative)', 'EOM(Positive)',
                    'EOM(Negative)', 'AUCRatio', 'OverAllAcc', 'OverAllAUC', 'TOTALACC']
LOWER_BETTER_COLS=['FPR', 'FNR', 'EOpp0', 'EOpp1','EBAcc',
                    'EOdd', 'EOddAbs','EOddMax','AUCDiff', 'TOTALACCDIF', 'ACCDIF']
PERF_COLS=['AUC', 'ACC', 'TPR', 'TNR', 'PPV', 'NPV','PR','NR','BAcc',
            'FPR', 'FNR', 'OverAllAcc', 'OverAllAUC', 'TOTALACC']
FAIRNESS_COLS=['PQD', 'PQD(class)', 'EPPV', 'ENPV', 'DPM(Positive)', 'DPM(Negative)', 'EOM(Positive)',
                'EOM(Negative)', 'AUCRatio', 'EOpp0', 'EOpp1','EBAcc', 'EOdd', 'EOddAbs','EOddMax','AUCDiff', 'TOTALACCDIF', 'ACCDIF']

# maps the csv name to the TCGA project name
TCGA_NAME_DICT = {
    # tumor detection
    'LUAD_TumorDetection':  '04_LUAD',
    'CCRCC_TumorDetection':  '06_KIRC',
    'HNSC_TumorDetection':  '07_HNSC',
    'LSCC_TumorDetection':  '10_LUSC',
    # 'BRCA_TumorDetection':  '01_BRCA',
    'PDA_TumorDetection':  '11_PRAD',
    'UCEC_TumorDetection':  '05_UCEC',
    # cancer type classification
    'COAD_READ_512': '_COAD+READ',
    'KIRC_KICH_512': '_KIRC+KICH',
    'KIRP_KICH_512': '_KIRP+KICH',
    'KIRC_KIRP_512': '_KIRC+KIRP',   
    'LGG_GBM_512': '_GBM+LGG',
    'LUAD_LUSC_512': '_LUAD+LUSC',
    'COAD_READ': '_COAD+READ',
    'KIRC_KICH': '_KIRC+KICH',
    'KIRP_KICH': '_KIRP+KICH',
    'KIRC_KIRP': '_KIRC+KIRP',   
    'LGG_GBM': '_GBM+LGG',
    'LUAD_LUSC': '_LUAD+LUSC',
    # cancer subtype classification
    'Breast_ductal_lobular_512': '01_BRCA 1+1',
    'LUAD_BRONCHIOLO-ALVEOLAR_512': '04_LUAD 3+n',
    'Breast_ductal_lobular': '01_BRCA 1+1',
    'LUAD_BRONCHIOLO-ALVEOLAR': '04_LUAD 3+n',
    'LUAD_3_n': '04_LUAD 3+n',
}

CUTOFF_METHODS=Literal[None,'none','MicroBAcc','MacroBAcc','MicroDistance','MacroDistance','MicroRecallDiff']

def calculate_xauc(y_true, y_score, groups, positive_label=1):
    """
    Calculates the cross-group Area Under the Curve (xAUC) disparity.

    The xAUC disparity measures the difference in the probability of correctly
    ranking a positive instance from one group above a negative instance from
    another group, and vice-versa. A value close to zero indicates fairness
    in ranking across the two groups.

    Args:
        y_true (np.ndarray): Array of true binary labels.
        y_score (np.ndarray): Array of predicted scores or probabilities.
        groups (np.ndarray): Array of group labels for each instance.
                             This function assumes two unique groups.
        positive_label (int or str, optional): The label of the positive
                                               class. Defaults to 1.

    Returns:
        float: The calculated xAUC disparity. Returns np.nan if the calculation
               is not possible for one of the cross-group pairs (e.g., if a
               group has no positive or no negative examples).
    """
    # Ensure inputs are numpy arrays for boolean indexing
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    groups = np.asarray(groups)

    # Identify the two unique groups in the data
    unique_groups = np.unique(groups)
    if len(unique_groups) != 2:
        raise ValueError(f"Expected 2 unique groups, but found {len(unique_groups)}.")
    
    group_a_label, group_b_label = unique_groups[0], unique_groups[1]
    
    # Identify positive and negative labels
    unique_labels = np.unique(y_true)
    if len(unique_labels) != 2:
        raise ValueError(f"Expected 2 unique labels in y_true, but found {len(unique_labels)}.")
    if positive_label not in unique_labels:
        raise ValueError(f"positive_label '{positive_label}' not found in y_true labels {unique_labels}.")
    
    negative_label = unique_labels[0] if unique_labels[1] == positive_label else unique_labels[1]

    # Create boolean masks for filtering data
    is_group_a = (groups == group_a_label)
    is_group_b = (groups == group_b_label)
    is_positive = (y_true == positive_label)
    is_negative = (y_true == negative_label)

    # --- Calculate AUC for (Positives from Group A, Negatives from Group B) ---
    y_true_ab = np.concatenate([y_true[is_group_a & is_positive], y_true[is_group_b & is_negative]])
    y_score_ab = np.concatenate([y_score[is_group_a & is_positive], y_score[is_group_b & is_negative]])

    try:
        # We need to pass labels to roc_auc_score in case y_true doesn't contain 0/1
        auc_ab = roc_auc_score(y_true_ab, y_score_ab, labels=[negative_label, positive_label])
    except ValueError:
        # This occurs if one class is not present (e.g., no positives in A or no negatives in B)
        auc_ab = np.nan
        print(f"Warning: Could not calculate AUC for positives from '{group_a_label}' and negatives from '{group_b_label}'.")


    # --- Calculate AUC for (Positives from Group B, Negatives from Group A) ---
    y_true_ba = np.concatenate([y_true[is_group_b & is_positive], y_true[is_group_a & is_negative]])
    y_score_ba = np.concatenate([y_score[is_group_b & is_positive], y_score[is_group_a & is_negative]])

    try:
        auc_ba = roc_auc_score(y_true_ba, y_score_ba, labels=[negative_label, positive_label])
    except ValueError:
        auc_ba = np.nan
        print(f"Warning: Could not calculate AUC for positives from '{group_b_label}' and negatives from '{group_a_label}'.")


    # --- Calculate and return the xAUC disparity ---
    if np.isnan(auc_ab) or np.isnan(auc_ba):
        return np.nan
    
    xauc_disparity = auc_ab - auc_ba
    return xauc_disparity


def groupwise_balanced_accuracy(y_true, y_pred, groups):
    '''
    Calculate recall for each group
    Args:
    * y_true: numpy array, ground truth labels
    * y_pred: numpy array, model predictions
    * groups: numpy array, group labels

    Returns:
    * recall: numpy array, recall for each group
    '''
    groups = np.array(groups)
    unique_groups = np.unique(groups)
    balanced_acc = []
    for group in unique_groups:
        idx = groups == group
        balanced_acc.append(sklearn.metrics.recall_score(y_true[idx], y_pred[idx],average='macro', zero_division=0))
    balanced_acc = np.stack(balanced_acc)
    return balanced_acc


# def macro_recall(y_true, y_pred, groups):

def Find_Optimal_Cutoff(
    target, probs,
    sensitives=None,
    method:CUTOFF_METHODS=None):
    """ Find the optimal probability cutoff point for a classification model related to event rate Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations
    probs : Matrix with class probabilities, where rows are observations
    sensitives : Matrix with sensitive data, where rows are observations
    method : str, method to find the optimal cutoff point. Options are:
        None: use the default threshold of 0.5
        'none': use the default threshold of 0.5
        'MicroBAcc': use the threshold that maximizes the micro balanced accuracy
        'MacroBAcc': use the threshold that maximizes the macro balanced accuracy (across groups)
    Returns
    ---------- 
    list type, with optimal cutoff value   
    """
    if method is None:
        method = 'none'
        
    # no adjustment
    if method == 'none':
        return 0.5
    
    # find the threshold that maximizes the micro balanced accuracy
    if method == 'MicroBAcc':
        fpr, tpr, thresholds = metrics.roc_curve(target, probs)
        threshold = thresholds[np.argmax(tpr - fpr)]
        return threshold
    
    # find the threshold that maximizes the macro balanced accuracy (across groups)
    if method == 'MacroBAcc':
        assert sensitives is not None, 'sensitives must be provided for MacroBAcc'
        baccs = []
        # thresholds = probs
        thresholds = np.concatenate([probs - 1e-10, np.expand_dims(np.max(probs), axis=0) + 1e-10])
        for threshold in thresholds:
            y_pred = probs > threshold
            group_bacc = groupwise_balanced_accuracy(target, y_pred, sensitives)
            macro_bacc = np.mean(group_bacc)
            baccs.append(macro_bacc)
        baccs = np.array(baccs)
        threshold = thresholds[np.argmax(baccs)]
        return threshold
    # find the threshold that minimizes the distance to the point (0,1) in the ROC curve
    if method == 'MicroDistance':
        fpr, tpr, thresholds = metrics.roc_curve(target, probs)
        x = 1 - tpr
        y = fpr
        d = np.sqrt(x**2 + y**2)
        idx_max = np.argmin(d)
        threshold = thresholds[idx_max]
        return threshold
    # find the threshold that minimizes the macro-averaged distance to the point (0,1) in the ROC curve between groups
    if method == 'MacroDistance':
        assert sensitives is not None, 'sensitives must be provided for MacroDistance'
        groups = np.array(sensitives)
        unique_groups = np.unique(groups)
        thresholds = np.concatenate([probs - 1e-10, np.expand_dims(np.max(probs), axis=0) + 1e-10])
        dists = np.zeros((len(unique_groups), len(thresholds)))
        for i_g,group in enumerate(unique_groups):
            idx = groups == group
            y_group = target[idx]
            prob_group = probs[idx]
            for i, threshold in enumerate(thresholds):
                pred_group = prob_group > threshold
                tpr = sklearn.metrics.recall_score(y_group, pred_group,zero_division=0)
                tnr = sklearn.metrics.recall_score(y_group, pred_group, pos_label=0,zero_division=0)
                x = 1 - tpr
                y = 1 - tnr
                d = np.sqrt(x**2 + y**2)
                dists[i_g, i] = d
        macro_dists = np.mean(dists, axis=0)
        idx_min = np.argmin(macro_dists)
        opt_threshold = thresholds[idx_min]
        return opt_threshold
    if method == 'MicroRecallDiff':
        fpr, tpr, thresholds = metrics.roc_curve(target, probs)
        threshold = thresholds[np.argmin(np.abs(tpr -  (1-fpr )))]
        return threshold

        
    
    raise notImplementedError(f'method {method} is not implemented')

def save_thresholds(args,target, probs,sensitives, outpath, 
                    save_metrics:bool=False,
                    save_metrics_postfix='',
                    log_method:CUTOFF_METHODS=None):
    df = pd.DataFrame({'method':[],'threshold':[]})
    for method in ['none','MicroBAcc','MacroBAcc','MicroDistance','MacroDistance']:
        threshold = Find_Optimal_Cutoff(target, probs, sensitives, method)
        # insert a new row
        df.loc[len(df)] = [method,threshold]
    outcsv = os.path.join(outpath,'thresholds.csv')
    df.to_csv(outcsv,index=False)
    if save_metrics:
        threshold = df.loc[df['method']==log_method]['threshold'].values[0]
        save_metrics_summary(args,target,probs, sensitives, outpath,postfix=save_metrics_postfix)



def load_thresholds(inpath, method:CUTOFF_METHODS=None):
    if method is None:
        method = 'none'
    csv = os.path.join(inpath,'thresholds.csv')
    df = pd.read_csv(csv)
    return df.loc[df['method']==method]['threshold'].values[0]


def get_predictions(probs, inpath, method:CUTOFF_METHODS=None):
    threshold = load_thresholds(inpath, method)
    predictions = np.where(probs > threshold, 1, 0)
    return predictions

def save_metrics_summary(args,labels,probs, senAttrs, result_path,postfix='',verbose=True):
    if postfix != "":
        postfix = "_" + postfix
    predictions = get_predictions(probs, result_path, args.cutoff_method)
    results = FairnessMetrics(predictions, probs, labels, senAttrs)
    csv_path = os.path.join(result_path, f"result{postfix}.csv")
    pd.DataFrame(results).T.to_csv(csv_path)
    if verbose:
        print(f"Save results to:{csv_path}")

    
        

    
def FairnessMetrics(predictions, probs, labels, sensitives,
                    previleged_group=None, unprevileged_group=None, add_perf_difference=False):
    '''
    Estimating fairness metrics
    Args:
    * predictions: numpy array, model predictions
    * probs: numpy array, model probabilities
    * labels: numpy array, ground truth labels
    * sensitives: numpy array, sensitive attributes
    * previleged_group: str, previleged group name. If None, the group with the best performance will be used.
    * unprevileged_group: str, unprevileged group name. If None, the group with the worst performance will be used.
    * add_perf_difference: bool, whether to add performance difference metrics

    Returns:
    * results: dict, performance metrics and fairness metrics
    '''
    AUC = []
    ACC = []
    TPR = []
    TNR = []
    PPV = []
    NPV = []
    PR = []
    NR = []
    FPR = []
    FNR = []
    TOTALACC = []
    N = []
    N_0 = []
    N_1 = []
    labels = labels.astype(np.int64)
    sensitives = [str(x) for x in sensitives]
    df = pd.DataFrame({'pred': predictions, 'prob': probs,
                      'label': labels, 'group': sensitives})

    uniSens = np.unique(sensitives)
    ## categorize the groups into majority and minority groups
    # group_types = []
    if previleged_group is not None:
        if unprevileged_group is None:
            # if only the previleged group is provided, we categorize the rest of the groups as unprevileged
            group_types = ['majority' if group == previleged_group else 'minority' for group in uniSens]
        else:
            # if both previleged and unprevileged groups are provided, we categorize the groups accordingly
            # for group in uniSens:
            #     g_type = 'majority' if group == previleged_group else 'minority' if group == unprevileged_group else 'unspecified'
            group_types = ['majority' if group == previleged_group else 'minority' if group == unprevileged_group else 'unspecified' for group in uniSens]
    elif unprevileged_group is not None:
        # if only the unprevileged group is provided, we categorize the rest of the groups as previleged
        group_types = ['minority' if group == unprevileged_group else 'majority' for group in uniSens]
        # for group in uniSens:
            # g_type = 'minority' if group == unprevileged_group else 'majority'
            # group_types.append(g_type)
    else:
        # if both previleged and unprevileged groups are not provided, set to unspecified
        group_types = ['unspecified' for group in uniSens]
        # for group in uniSens:
        #     g_type =  'unspecified'
        #     group_types.append(g_type)
                
    
            
            
            
            
    ##
    
    for modeSensitive in uniSens:
        modeSensitive = str(modeSensitive)
        df_sub = df.loc[df['group'] == modeSensitive]
        y_pred = df_sub['pred'].to_numpy()
        y_prob = df_sub['prob'].to_numpy()
        y_true = df_sub['label'].to_numpy()
        # y_pred = predictions[sensitives == modeSensitive]
        # y_prob = probs[sensitives == modeSensitive]
        # y_true = labels[sensitives == modeSensitive]

        if len(y_pred) == 0:
            continue
        cnf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
        CR = classification_report(y_true, y_pred, labels=[
                                   0, 1], output_dict=True, zero_division=0)
        # AUC
        if len(set(y_true)) == 1:
            AUC.append(np.nan)
        else:
            AUC.append((metrics.roc_auc_score(y_true, y_prob)))
        N.append(CR['macro avg']['support'])
        N_0.append(CR['0']['support'])
        N_1.append(CR['1']['support'])
        # Overall accuracy for each class
        ACC.append(np.trace(cnf_matrix)/np.sum(cnf_matrix))
        # Sensitivity, hit rate, recall, or true positive rate
        TPR.append(CR['1']['recall'] if CR['1']['support'] > 0 else np.nan)
        # Specificity or true negative rate
        TNR.append(CR['0']['recall'] if CR['0']['support'] > 0 else np.nan)
        # Precision or positive predictive value
        PPV.append(CR['1']['precision'] if np.sum(
            cnf_matrix[:, 1]) > 0 else np.nan)
        # Negative predictive value
        NPV.append(CR['0']['precision'] if np.sum(
            cnf_matrix[:, 0]) > 0 else np.nan)
        # Fall out or false positive rate
        FPR.append(1-CR['0']['recall'] if CR['0']['support'] > 0 else np.nan)
        # False negative rate
        FNR.append(1-CR['1']['recall'] if CR['1']['support'] > 0 else np.nan)
        # Prevalence
        PR.append(np.sum(cnf_matrix[:, 1]) / np.sum(cnf_matrix))
        # Negative Prevalence
        NR.append(np.sum(cnf_matrix[:, 0]) / np.sum(cnf_matrix))
        # # False discovery rate
        # FDR = FP/(TP+FP)
        # total ACC
        TOTALACC.append(np.trace(cnf_matrix)/np.sum(cnf_matrix))

    OverAll_cnf_matrix = confusion_matrix(predictions, labels)
    OverAllACC = np.trace(OverAll_cnf_matrix)/np.sum(OverAll_cnf_matrix)
    try:
        OverAllAUC = metrics.roc_auc_score(labels, probs)
    except:
        OverAllAUC = np.nan

    df_perf = pd.DataFrame(
        {'AUC': AUC, 'ACC': ACC, 'TPR': TPR, 'TNR': TNR, 'PPV': PPV, 'NPV': NPV, 'BAcc': (np.array(TPR)+np.array(TNR))/2,
         'PR': PR, 'NR': NR, 'FPR': FPR, 'FNR': FNR, 'TOTALACC': TOTALACC,'OverAllAcc': OverAllACC,
         'Odd1': np.array(TPR)+np.array(FPR),'Odd0':np.array(TNR)+np.array(FNR),
         'OverAllAUC': OverAllAUC}, index=uniSens)
    lower_better_metrics = ['FPR', 'FNR']
    higher_better_metrics = ['TPR', 'TNR', 'NPV','BAcc',
                             'PPV', 'TOTALACC','OverAllAcc','OverAllAUC', 'AUC', 'ACC', 'PR', 'NR','Odd1','Odd0']
    
    if previleged_group is not None:
        perf_previleged = df_perf.loc[previleged_group]
    else:
        perf_previleged = pd.concat([
            df_perf[higher_better_metrics].max(),
            df_perf[lower_better_metrics].min()])
    if unprevileged_group is not None:
        perf_unprevileged = df_perf.loc[unprevileged_group]
    elif previleged_group is not None:
        perf_not_previleged = df_perf.drop(
            previleged_group)
        # perf_unprevileged = perf_not_previleged.min()
        perf_unprevileged = pd.concat([
            perf_not_previleged[higher_better_metrics].min(),
            perf_not_previleged[lower_better_metrics].max()])
    else:
        # perf_unprevileged = df_perf[higher_better_metrics].min()
        perf_unprevileged = pd.concat([
            df_perf[higher_better_metrics].min(),
            df_perf[lower_better_metrics].max()])

    perf_diff = perf_previleged - perf_unprevileged
    # perf_diff[lower_better_metrics] = perf_unprevileged[lower_better_metrics] - perf_previleged[lower_better_metrics]
    perf_ratio = perf_unprevileged / perf_previleged
    # perf_ratio[lower_better_metrics] = perf_previleged[lower_better_metrics] / perf_unprevileged[lower_better_metrics]

    AUC = np.array(AUC)
    ACC = np.array(ACC)
    TPR = np.array(TPR)
    TNR = np.array(TNR)
    PPV = np.array(PPV)
    NPV = np.array(NPV)
    PR = np.array(PR)
    NR = np.array(NR)
    FPR = np.array(FPR)
    FNR = np.array(FNR)
    TOTALACC = np.array(TOTALACC)
    BAcc = (TPR+TNR)/2
    
    results = {
        'sensitiveAttr': uniSens,
        'group_type': group_types,
        'N_0': N_0,
        'N_1': N_1,
        'AUC': AUC,
        'ACC': ACC,
        'TPR': TPR,
        'TNR': TNR,
        'PPV': PPV,
        'NPV': NPV,
        'BAcc': BAcc,
        'PR': PR,
        'NR': NR,
        'FPR': FPR,
        'FNR': FNR,
        'EOpp0': perf_diff['TNR'],
        'EOpp1': perf_diff['TPR'],
        'EBAcc': perf_diff['BAcc'],
        'EOdd':  (-perf_diff['Odd1']), # equivalent to aif360.metrics.average_odds_difference()
        'EOddAbs': (np.abs(perf_diff['TNR']) + np.abs(perf_diff['TPR']))/2, # equivalent to aif360.metrics.average_abs_odds_difference()
        'EOddMax':  np.max([np.abs(perf_diff['TNR']), np.abs(perf_diff['TPR'])]),   # equivalent to aif360.metrics.equalized_odds_difference()
        'PQD': perf_ratio['TOTALACC'],
        'PQD(class)': perf_ratio['TOTALACC'],
        'EPPV': perf_ratio['PPV'],
        'ENPV': perf_ratio['NPV'],
        'DPM(Positive)': perf_ratio['PR'],
        'DPM(Negative)': perf_ratio['NR'],
        'EOM(Positive)': perf_ratio['TPR'],
        'EOM(Negative)':  perf_ratio['TNR'],
        'AUCRatio':  perf_ratio['AUC'],
        'AUCDiff':  perf_diff['AUC'],
        'OverAllAcc': OverAllACC,
        'OverAllAUC': OverAllAUC,
        'TOTALACC': TOTALACC,
        'TOTALACCDIF': perf_diff['TOTALACC'],
        'ACCDIF': perf_diff['ACC'],
    }
    if add_perf_difference:
        results, new_cols = get_perf_diff(
            results,perf_metrics=PERF_COLS,
            privileged_group=previleged_group,demo_col='sensitiveAttr')
    return results

def get_metric_names(add_perf_difference=False):
    '''
    returns a dictionary of metric list
    '''
    metrics_list = {
        'higher_better_metrics': HIGHER_BETTER_COLS.copy(),
        'lower_better_metrics': LOWER_BETTER_COLS.copy(),
        'perf_metrics': PERF_COLS.copy(),
        'fairness_metrics': FAIRNESS_COLS.copy()
    }
    ## if add_perf_difference == True, we add the performance difference as a fairness metric
    if add_perf_difference:
        for col in PERF_COLS:
            metrics_list['fairness_metrics'].append(f'{col}_diff')
            if col in HIGHER_BETTER_COLS:
                metrics_list['lower_better_metrics'].append(f'{col}_diff')
            elif col in LOWER_BETTER_COLS:
                metrics_list['higher_better_metrics'].append(f'{col}_diff')
    metrics_list['higher_better_fairness_metrics'] = [x for x in metrics_list['fairness_metrics'] if x in metrics_list['higher_better_metrics']]
    metrics_list['lower_better_fairness_metrics'] = [x for x in metrics_list['fairness_metrics'] if x in  metrics_list['lower_better_metrics']]
    return metrics_list

# for inpath in BASES:
def get_perf_diff(fairResult,perf_metrics=PERF_COLS,privileged_group=None,demo_col = 'sensitiveAttr'):
    '''
    Add performance metric to the dataframe
    '''
    df = pd.DataFrame(fairResult)
    new_cols = []
    for col in perf_metrics:
        if col in df.keys():
            if privileged_group is not None:
                val_privileged = df.loc[df[demo_col]==privileged_group][col]
                val_others = df.loc[df[demo_col]!=privileged_group][col]
            else:
                val_privileged = df[col]
                val_others = df[col]
            val_privileged = val_privileged.max()
            val_others = val_others.min()
            
            
            # df[f'{col}Diff'] = val_privileged - val_others
            fairResult[col+'_diff'] = val_privileged - val_others
            new_cols.append(col+'_diff')
            
    return fairResult, new_cols

