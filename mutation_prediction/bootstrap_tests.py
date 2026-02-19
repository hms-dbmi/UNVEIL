"""
Bootstrap significance testing for fairness evaluation.

This module consolidates bootstrap testing functions from the bootstrap_significant_test
directory into a single, well-organized module.
"""

import pandas as pd
import numpy as np
from scipy.stats import combine_pvalues
from scipy import stats
from sklearn.metrics import roc_auc_score
from typing import List, Tuple, Literal
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing
from functools import partial

# Import from fairness_utils
from fairness_utils import FairnessMetrics

# Metric name constants
HIGHER_BETTER_COLS=['AUC', 'ACC', 'TPR', 'TNR', 'PPV', 'NPV', 'PQD', 'PQD(class)','PR','NR','BAcc',
                    'EPPV', 'ENPV', 'DPM(Positive)', 'DPM(Negative)', 'EOM(Positive)',
                    'EOM(Negative)', 'AUCRatio', 'OverAllAcc', 'OverAllAUC', 'TOTALACC']
LOWER_BETTER_COLS=['FPR', 'FNR', 'EOpp0', 'EOpp1','EBAcc',
                    'EOdd', 'EOddAbs','EOddMax','AUCDiff', 'TOTALACCDIF', 'ACCDIF']
PERF_COLS=['AUC', 'ACC', 'TPR', 'TNR', 'PPV', 'NPV','PR','NR','BAcc',
            'FPR', 'FNR', 'OverAllAcc', 'OverAllAUC', 'TOTALACC']
FAIRNESS_COLS=['PQD', 'PQD(class)', 'EPPV', 'ENPV', 'DPM(Positive)', 'DPM(Negative)', 'EOM(Positive)',
                'EOM(Negative)', 'AUCRatio', 'EOpp0', 'EOpp1','EBAcc', 'EOdd', 'EOddAbs','EOddMax','AUCDiff', 'TOTALACCDIF', 'ACCDIF']


def get_metric_names(add_perf_difference=False):
    """Get metric names organized by category."""
    METRIC_NAMES_DICT = {
        'perf_metrics': PERF_COLS,
        'fairness_metrics': FAIRNESS_COLS,
        'lower_better_fairness_metrics': LOWER_BETTER_COLS,
        'higher_better_fairness_metrics': HIGHER_BETTER_COLS,
    }
    if add_perf_difference:
        METRIC_NAMES_DICT['fairness_metrics'] = FAIRNESS_COLS + [f'{m}_diff' for m in PERF_COLS]
    return METRIC_NAMES_DICT

def bias_bootstrap_iteration(i, df, df_g, privileged_group, add_perf_difference,group_col=None):
    """Performs a single bootstrap iteration."""
    
    if group_col is None:
        df_shuffled = df.sample(
            frac=1, replace=True).reset_index(drop=True)  # shuffle the data (bootstrap with replacement)
    else: #if group_col is not None, we sample with replacement within each group
        df_shuffled = df.groupby(group_col).apply(lambda x: x.sample(frac=1, replace=True)).reset_index(drop=True)
        
    df_shuffled['sens_attr'] = df_g  # add the sensitive attribute
    # get the estimated fairness metrics for the bootstrap sample
    fairResult_bootstrap = FairnessMetrics(
        df_shuffled[f'pred'].to_numpy(),
        df_shuffled[f'prob'].to_numpy(),
        df_shuffled['label'].to_numpy(),
        df_shuffled['sens_attr'].astype(str).to_numpy(),
        privileged_group=privileged_group,add_perf_difference=add_perf_difference)
    
    return fairResult_bootstrap
            
def bootstrap_bias_test(df,
                        bootstrap_n=1000, privileged_group=None,add_perf_difference=True,group_col=None,verbose=True):
    '''
    Estimate the p-value of the fairness metrics using bootstrap
    Args:
    * df: pandas DataFrame, data, including the following columns:
        * sens_attr: sensitive attributes
        * prob: model probabilities
        * label: ground truth labels
        * pred:  model predictions
    * bootstrap_n: int, number of bootstrap samples
    * privileged_group: str, previleged group name. If None, the group with the best performance will be used.
    * group_col: if not None, the column to group by for bootstrap sampling. If None, we sample the whole dataframe.
    Returns:
    * fairResult: dict, performance metrics and fairness metrics
    * p_table_worse: pandas Series, p-value of the fairness metrics (biased against the unprevileged group)
    * p_table_better: pandas Series, p-value of the fairness metrics (biased in favor of the unprevileged group)
    '''
    METRIC_NAMES_DICT = get_metric_names(add_perf_difference=add_perf_difference)
    if group_col is None:
        df = df[['sens_attr', 'prob', 'label', 'pred']]
        df_g = df['sens_attr']  # sensitive attribute
        df_val = df[['prob', 'label', 'pred']]  # prediction and label
    else:
        df = df[['sens_attr', 'prob', 'label', 'pred', group_col]]
        df_g = df['sens_attr']
        df_val = df[['prob', 'label', 'pred', group_col]]  # prediction and label
    
    ########################################################
    # get the estimated fairness metrics
    fairResult = FairnessMetrics(
        df[f'pred'].to_numpy(),
        df[f'prob'].to_numpy(),
        df['label'].to_numpy(),
        df['sens_attr'].astype(str).to_numpy(),
        privileged_group=privileged_group,add_perf_difference=add_perf_difference)

    ########################################################
    # get the estimated fairness metrics
    if verbose:
        df_temp = pd.DataFrame(fairResult)
        df_perf_temp = pd.DataFrame(df_temp[METRIC_NAMES_DICT['perf_metrics']])
        df_fair_temp = pd.DataFrame(df_temp[METRIC_NAMES_DICT['fairness_metrics']])
        pd.set_option('display.max_rows', df_temp.shape[1]) 
        pd.set_option('display.precision', 3)
        df_disp = pd.concat([df_perf_temp.T.reset_index(),df_fair_temp.T.reset_index()],axis=1)
        print('='*75)
        print(f'\t\tEstimated performance & fairness metrics')
        print('='*75)
        print(df_disp)
        print('='*75)

        # print('='*100)  
        # print(f'Estimated performance metrics:')
        # print(df_perf_temp.T)
        # print('='*100)  

        # print(f'Estimated fairness metrics:')
        # print(df_fair_temp.T)


    ########################################################
    # ########################################################
    # # get bootstrap samples of the fairness metrics
    # bootstrap_results = []
    # for i in tqdm(range(bootstrap_n),miniters=bootstrap_n//10):
    #     df_shuffled = df_val.sample(
    #         frac=1, replace=True).reset_index(drop=True)  # shuffle the data (bootstrap with replacement)
    #     df_shuffled['sens_attr'] = df_g  # add the sensitive attribute
    #     # get the estimated fairness metrics for the bootstrap sample
    #     fairResult_bootstrap = FairnessMetrics(
    #         df_shuffled[f'pred'].to_numpy(),
    #         df_shuffled[f'prob'].to_numpy(),
    #         df_shuffled['label'].to_numpy(),
    #         df_shuffled['sens_attr'].astype(str).to_numpy(),
    #         privileged_group=privileged_group,add_perf_difference=add_perf_difference)
    #     # fairResult = fairResult

    #     bootstrap_results.append(fairResult_bootstrap)
        
    def parallelized_bootstrap(bootstrap_n, df, df_g, privileged_group, add_perf_difference, METRIC_NAMES_DICT,group_col=None):
        """Parallelizes the bootstrap process using multiprocessing.Pool."""

        bootstrap_results = []

        with Pool(4) as pool:
            pool_func = partial(bias_bootstrap_iteration, df=df, df_g=df_g, privileged_group=privileged_group, add_perf_difference=add_perf_difference, group_col=group_col)
            bootstrap_results = list(tqdm(pool.imap(pool_func, range(bootstrap_n)), total=bootstrap_n, miniters=bootstrap_n//10))
            

        df_bootstrap = pd.DataFrame.from_records(bootstrap_results)[METRIC_NAMES_DICT['fairness_metrics']]
        return df_bootstrap
    
    df_bootstrap = parallelized_bootstrap(bootstrap_n, df, df_g, privileged_group, add_perf_difference, METRIC_NAMES_DICT,group_col=group_col)
        
    # df_bootstrap = pd.DataFrame.from_records(bootstrap_results)[METRIC_NAMES_DICT['fairness_metrics']]
    ########################################################
    # get the percentile of the measured values over bootstraped samples
    # p-value of the fairness metrics (biased against the unprevileged group)
    p_table_worse = {}
    # p-value of the fairness metrics (biased in favor of the unprevileged group)
    p_table_better = {}
    for metric in METRIC_NAMES_DICT['fairness_metrics']:
        measure = fairResult[metric]
        n_valid_bootstrap = df_bootstrap[metric].notna().sum()
        if n_valid_bootstrap < bootstrap_n:
            print(
                f'Warning: {n_valid_bootstrap} valid bootstrap samples for {metric}. Expected {bootstrap_n}')
        if metric in METRIC_NAMES_DICT['lower_better_fairness_metrics']:
            p_table_worse[metric] = (df_bootstrap[metric] >=
                                     measure).sum() / n_valid_bootstrap
            p_table_better[metric] = (df_bootstrap[metric] <=
                                      measure).sum() / n_valid_bootstrap
        else:
            p_table_worse[metric] = (df_bootstrap[metric] <=
                                     measure).sum() / n_valid_bootstrap
            p_table_better[metric] = (df_bootstrap[metric] >=
                                      measure).sum() / n_valid_bootstrap
    p_table_worse = pd.Series(
        p_table_worse)
    p_table_better = pd.Series(
        p_table_better)
    fairResult = pd.DataFrame(fairResult)
    p_table_worse = pd.DataFrame(p_table_worse, columns=['pval']).transpose()
    p_table_better = pd.DataFrame(p_table_better, columns=['pval']).transpose()
    # fairResult = pd.DataFrame(fairResult).transpose()
    return fairResult, p_table_worse, p_table_better

def CI_bootstrap_iteration(i, df, df_g, privileged_group, add_perf_difference,group_col=None):
    """Performs a single bootstrap iteration."""
    # Sample with replacement within each group for confidence intervals
    if group_col is None:
        # df_shuffled = df.groupby('sens_attr').apply(sample_with_replacement).reset_index(drop=True)
        df_shuffled = df.groupby('sens_attr').sample(frac=1, replace=True).reset_index(drop=True)
        df_shuffled['sens_attr'] = df_g  # add the sensitive attribute
    else:
        # df_shuffled = df.groupby([group_col,'sens_attr']).apply(sample_with_replacement)#.reset_index(drop=True)
        df_shuffled = df.groupby([group_col,'sens_attr']).sample(frac=1, replace=True).reset_index(drop=True)
        # df_shuffled['sens_attr'] = df_g

    # Calculate fairness metrics
    fairResult_bootstrap = FairnessMetrics(
        df_shuffled[f'pred'].to_numpy(),
        df_shuffled[f'prob'].to_numpy(),
        df_shuffled['label'].to_numpy(),
        df_shuffled['sens_attr'].astype(str).to_numpy(),
        privileged_group=privileged_group,
        add_perf_difference=add_perf_difference
    )
    return fairResult_bootstrap
def bootstrap_bias_CI(df,bootstrap_n=1000, privileged_group=None,add_perf_difference=True,group_col=None):
    
    '''
    Estimate the confidence interval of the fairness metrics using bootstrap
    Args:
    * df: pandas DataFrame, data, including the following columns:
        * sens_attr: sensitive attributes
        * prob: model probabilities
        * label: ground truth labels
        * pred:  model predictions
    * bootstrap_n: int, number of bootstrap samples
    * privileged_group: str, previleged group name. If None, the group with the best performance will be used.
    Returns:
    '''
    METRIC_NAMES_DICT = get_metric_names(add_perf_difference=add_perf_difference)

    if group_col is None:
        df = df[['sens_attr', 'prob', 'label', 'pred']]
        df_g = df['sens_attr']  # sensitive attribute
        df_val = df[['prob', 'label', 'pred']]  # prediction and label
    else:
        df = df[['sens_attr', 'prob', 'label', 'pred', group_col]]
        df_g = df['sens_attr']
        df_val = df[['prob', 'label', 'pred', group_col]]  # prediction and label
    
    ########################################################
    # get the estimated fairness metrics
    fairResult = FairnessMetrics(
        df[f'pred'].to_numpy(),
        df[f'prob'].to_numpy(),
        df['label'].to_numpy(),
        df['sens_attr'].astype(str).to_numpy(),
        privileged_group=privileged_group,add_perf_difference=add_perf_difference)


    def parallelized_bootstrap(bootstrap_n, df, df_g, privileged_group, add_perf_difference, METRIC_NAMES_DICT,group_col=None):
        """Parallelizes the bootstrap process using multiprocessing.Pool."""

        bootstrap_results = []

        with Pool(4) as pool:
            pool_func = partial(CI_bootstrap_iteration, df=df, df_g=df_g, privileged_group=privileged_group, add_perf_difference=add_perf_difference,group_col=group_col)
            bootstrap_results = list(tqdm(pool.imap(pool_func, range(bootstrap_n)), total=bootstrap_n, miniters=bootstrap_n//10))
            

        df_bootstrap = pd.DataFrame.from_records(bootstrap_results)[METRIC_NAMES_DICT['fairness_metrics']]
        return df_bootstrap

    # Example usage:
    df_bootstrap = parallelized_bootstrap(bootstrap_n, df, df_g, privileged_group, add_perf_difference, METRIC_NAMES_DICT,group_col=group_col)
    
    #########################################################
    # get the confidence interval
    df_lb = df_bootstrap.quantile(0.025)
    df_ub = df_bootstrap.quantile(0.975)
    df_lb.name='CI_lb'
    df_ub.name='CI_ub'
        
    df_CI = pd.concat([df_lb,df_ub],axis=1).T
    # fairResult = pd.DataFrame(fairResult).transpose()
    # return fairResult, p_table_worse, p_table_better
    return df_CI
def CV_bootstrap_bias_test(
    dfs, privileged_group=None, n_bootstrap=1000,aggregate_method='fisher'):
    '''
    Estimate the improvement of the corrected model over the baseline model for all folds
    Input:
        dfs: list of pd.DataFrame, the results for each fold. Each pd.DataFrame contains the following columns:
            * sens_attr: sensitive attributes
            * prob: model probabilities
            * label: ground truth labels
            * pred:  model predictions
        n_bootstrap: int, number of bootstrap iterations
        aggregate_method: str, method to aggregate p-values. Options are:
            - 'concatenate': concatenate the input data and run a single statistical test
            - 'fisher', 'pearson', 'tippett', 'stouffer', 'mudholkar_george': methods to combine p-values, see scipy.stats.combine_pvalues for details
            - 'groupwise': estimate the fairness metrics first, and then perform bootstraping on population level
    Output:
        df_p_better: pd.DataFrame, the p-values for significant improvement
        df_p_worse: pd.DataFrame, the p-values for significant worsening
        
    '''
    
    if aggregate_method == 'concatenate':
        # if the method is concatenate, we concatenate the data and return a single p-value
        df = pd.concat(dfs).reset_index(drop=True)
        fairResult, df_p_worse, df_p_better = bootstrap_bias_test(df,
                        bootstrap_n=n_bootstrap, privileged_group=privileged_group)
        df_CI = bootstrap_bias_CI(df,bootstrap_n=n_bootstrap, privileged_group=privileged_group)

        return df_p_worse, df_p_better, fairResult, df_CI

        
    elif aggregate_method == 'fold_micro':
        ## perform bootstrap withing each fold, but test on the micro-averaged results
        for i in range(len(dfs)):
            dfs[i]['fold'] = i
        df = pd.concat(dfs).reset_index(drop=True)
        fairResult, df_p_worse, df_p_better = bootstrap_bias_test(df,
                        bootstrap_n=n_bootstrap, privileged_group=privileged_group,group_col='fold')
        df_CI = bootstrap_bias_CI(df,bootstrap_n=n_bootstrap, privileged_group=privileged_group,group_col='fold')

        return df_p_worse, df_p_better, fairResult, df_CI
    elif aggregate_method in ['fisher', 'pearson', 'tippett', 'stouffer', 'mudholkar_george']:
        # if the method is fisher or stouffer, we calculate the p-value for each fold
        dfs_p_better = []
        dfs_p_worse = []
        fairResult_list = []
        dfs_CI = []
        for i, df in enumerate(dfs):
            df_CI = bootstrap_bias_CI(df,bootstrap_n=n_bootstrap, privileged_group=privileged_group)
            fairResult, df_p_worse, df_p_better = bootstrap_bias_test(df,
                            bootstrap_n=n_bootstrap, privileged_group=privileged_group)
            df_p_better.insert(0, 'fold', i)
            df_p_worse.insert(0, 'fold', i)
            fairResult.insert(0, 'fold', i)
            dfs_CI.insert(0,df_CI)
            dfs_p_better.append(df_p_better)
            dfs_p_worse.append(df_p_worse)
            fairResult_list.append(fairResult)
            dfs_CI.append(df_CI)
        ## concatenate the p-values
        df_p_better = pd.concat(dfs_p_better)
        df_p_worse = pd.concat(dfs_p_worse)
        fairResult = pd.concat(fairResult_list)
        df_CI = pd.concat(dfs_CI)
        ## aggregate the p-values
        df_p_combined = dfs_p_better[0].copy()
        for i, row in df_p_combined.iterrows():
            for col in df_p_combined.columns:
                pvals = df_p_better[col].loc[i]
                meta_res = combine_pvalues(pvals, method=aggregate_method, weights=None)
                df_p_combined[col].loc[i] = meta_res.pvalue
        df_p_combined['fold'] = f'{aggregate_method}_combined'
        df_p_better = pd.concat([df_p_better, df_p_combined])
        
        df_p_combined = dfs_p_worse[0].copy()
        for i, row in df_p_combined.iterrows():
            for col in df_p_combined.columns:
                pvals = df_p_worse[col].loc[i]
                meta_res = combine_pvalues(pvals, method=aggregate_method, weights=None)
                df_p_combined[col].loc[i] = meta_res.pvalue
        df_p_combined['fold'] = f'{aggregate_method}_combined'
        df_p_worse = pd.concat([df_p_worse, df_p_combined])
        return df_p_worse, df_p_better, fairResult, df_CI

    elif aggregate_method == 'groupwise':
        # if the method is groupwise, we estimate the fairness metrics first, and then perform bootstraping on population level
        df_perf_avgdiff, df_p_worse, df_p_better, df_CI = bootstrap_bias_test_groupLevel(dfs,
                            bootstrap_n=n_bootstrap, privileged_group=privileged_group)

        return df_p_worse, df_p_better, df_perf_avgdiff, df_CI
    else:
        raise ValueError(
            f'Unknown aggregate method: {aggregate_method}. Available methods are: concatenate, fisher, pearson, tippett, stouffer, mudholkar_george, groupwise')

def get_mean_perf_diff(df_perf,add_perf_difference=True):
    METRIC_NAMES_DICT = get_metric_names(add_perf_difference=add_perf_difference)
    dfs_group = {group: df_perf.groupby(['group_type']).get_group(group) for group in ['majority','minority']}
    df_diff = dfs_group['majority'][METRIC_NAMES_DICT['perf_metrics']] - dfs_group['minority'][METRIC_NAMES_DICT['perf_metrics']]
    # for metrics that are lower the better, we reverse the sign
    lower_better_perf = list(set(METRIC_NAMES_DICT['lower_better_metrics']).intersection(set(df_diff.columns)))
    # df_diff[lower_better_perf] = -df_diff[lower_better_perf]
    return df_diff.mean(), df_diff
    
def bootstrap_bias_test_groupLevel(dfs,
                        bootstrap_n=1000, privileged_group=None,add_perf_difference=True):
    '''
    Do the bootstrap test on the group level (bootstrapping on gorup-level performance metrics)
    H0: mean performance difference between the unprivileged group and the privileged group (across all folds) is 0 
    Ha: > 0
    '''
    METRIC_NAMES_DICT = get_metric_names(add_perf_difference=add_perf_difference)

    ## get the performance metrics for each fold
    fairResult_list = []
    for i, df in enumerate(dfs):
        fairResult = FairnessMetrics(
            df[f'pred'].to_numpy(),
            df[f'prob'].to_numpy(),
            df['label'].to_numpy(),
            df['sens_attr'].astype(str).to_numpy(),
            privileged_group=privileged_group,add_perf_difference=True)
        fairResult['fold'] = i
        fairResult = pd.DataFrame(fairResult)
        fairResult_list.append(fairResult)
    # df = pd.DataFrame.from_records(fairResult_list)
    df = pd.concat(fairResult_list)
    df_perf = df[['sensitiveAttr','group_type','fold'] + METRIC_NAMES_DICT['perf_metrics']]
    df_perf = df_perf.set_index(['fold'],drop=True)
    df_perf_groupwise = df_perf.pivot(columns='sensitiveAttr').reset_index(drop=True)
    df_perf_groupwise.drop('group_type',axis=1,inplace=True)
    df_perf_mean = df_perf_groupwise.mean(axis=0)
    ## get performance difference (positive is biased against the unprivileged group)
    ## 
    df_perf_avgdiff,df_perf_diff = get_mean_perf_diff(df_perf)
    
    ## get CI for the performance difference
    dfs_CI = []
    dfs_perf_CI = []
    for i in tqdm(range(bootstrap_n),miniters=bootstrap_n//10):
        df_shuffled = df_perf_diff.sample(frac=1,replace=True).mean()
        df_perf_shuffled = df_perf_groupwise.sample(frac=1,replace=True).mean(axis=0)
        dfs_CI.append(df_shuffled)
        dfs_perf_CI.append(df_perf_shuffled)
        
    df_bootstrap = pd.concat(dfs_CI,axis=1).T   
    df_perf_bootstrap = pd.concat(dfs_perf_CI,axis=1).T
    df_CI = df_bootstrap.quantile([0.025,0.975])
    # df_CI.index.name = 'CI'
    # df_CI = df_CI.reset_index()
    df_CI.columns = [f'{x}_diff' if x != 'CI' else x for x in df_CI.columns]
    df_CI.loc['CI'] = [[df_CI.loc[0.025,col], df_CI.loc[0.975,col]] for col in df_CI.columns]
    df_CI = df_CI.loc[['CI']].reset_index(drop=True)
    ##
    df_perf_CI = df_perf_bootstrap.quantile([0.025,0.975])
    row_combined = {}
    for col in df_perf_CI.columns:
        row_combined[col] = [df_perf_CI.loc[0.025,col], df_perf_CI.loc[0.975,col]]
    row_combined = pd.Series(row_combined)
    df_perf_CI.loc['CI'] = row_combined
    df_perf_CI = df_perf_CI.loc[['CI']].reset_index(drop=True)
    df_perf_CI = df_perf_CI.melt()
    df_perf_CI.rename(columns={None:'metric','value':'CI'},inplace=True)
    df_perf_CI = df_perf_CI.pivot(columns='metric',index='sensitiveAttr',values='CI').reset_index()
    ##
    df_CI = pd.concat([df_CI]*df_perf_CI.shape[0],axis=0).reset_index(drop=True)
    df_CI = df_perf_CI.merge(df_CI,left_index=True,right_index=True)
    
    
    ## get the bootstrap samples
    bootstrap_results = []
    for i in tqdm(range(bootstrap_n),miniters=bootstrap_n//10):
        # bootstrap within each fold
        df_shuffled = df_perf.groupby('fold').sample(frac=1,replace=True)#.reset_index(drop=True)
        df_shuffled[['sensitiveAttr','group_type']] = df_perf[['sensitiveAttr','group_type']]  # add the sensitive attribute
        df_shuffled_avgdiff,_ = get_mean_perf_diff(df_shuffled)
        bootstrap_results.append(df_shuffled_avgdiff)
    df_bootstrap = pd.concat(bootstrap_results,axis=1).T
    
    ########################################################
    # get the percentile of the measured values over bootstraped samples
    # p-value of the fairness metrics (biased against the unprevileged group)
    p_table_worse = {}
    # p-value of the fairness metrics (biased in favor of the unprevileged group)
    p_table_better = {}
    for metric in METRIC_NAMES_DICT['perf_metrics']:
        measure = df_perf_avgdiff[metric]
        n_valid_bootstrap = df_bootstrap[metric].notna().sum()
        if n_valid_bootstrap < bootstrap_n:
            print(
                f'Warning: {n_valid_bootstrap} valid bootstrap samples for {metric}. Expected {bootstrap_n}')
        if metric in METRIC_NAMES_DICT['higher_better_metrics']:
            p_table_worse[metric] = (df_bootstrap[metric] >=
                                     measure).sum() / n_valid_bootstrap
            p_table_better[metric] = (df_bootstrap[metric] <=
                                      measure).sum() / n_valid_bootstrap
        else:
            p_table_worse[metric] = (df_bootstrap[metric] <=
                                     measure).sum() / n_valid_bootstrap
            p_table_better[metric] = (df_bootstrap[metric] >=
                                      measure).sum() / n_valid_bootstrap
    p_table_worse = pd.Series(
        p_table_worse)
    p_table_better = pd.Series(
        p_table_better)

    p_table_worse = pd.DataFrame(p_table_worse, columns=['pval']).transpose()
    p_table_better = pd.DataFrame(p_table_better, columns=['pval']).transpose()
    # fairResult = pd.DataFrame(fairResult).transpose()
    
        # df_shuffled = df_perf.sample(
        #     frac=1, replace=True).reset_index(drop=True)
    df_perf_avgdiff = df_perf_avgdiff.to_frame().T
    df_perf_avgdiff.columns = [f'{x}_diff' for x in df_perf_avgdiff.columns]
    p_table_worse.columns = [f'{x}_diff' for x in p_table_worse.columns]
    p_table_better.columns = [f'{x}_diff' for x in p_table_better.columns]
    return fairResult, p_table_worse, p_table_better, df_CI
def fairness_improvement(
        df_baseline,
        df_corrected,
        privileged_group,
        add_perf_difference=True
):
    '''
    calculate the improvement of the corrected model over the baseline model
    input:
        df_baseline: pd.DataFrame, the baseline results
        df_corrected: pd.DataFrame, the corrected results
        privileged_group: str, the privileged group
    output:
        improvement: float, the improvement
    '''
    metrics_list = []
    METRIC_NAMES_DICT = get_metric_names(add_perf_difference=add_perf_difference)

    for df in [df_baseline, df_corrected]:
        metrics = FairnessMetrics(
            df[f'pred'].to_numpy(),
            df[f'prob'].to_numpy(),
            df['label'].to_numpy(),
            df['sens_attr'].astype(str).to_numpy(), privileged_group=privileged_group,add_perf_difference=add_perf_difference)

        metrics_list.append(pd.DataFrame(
            metrics).set_index(['sensitiveAttr','group_type'], drop=True))
    improvement_larger_better = metrics_list[1] - metrics_list[0]
    improvement_smaller_better = metrics_list[0] - metrics_list[1]
    improvement = improvement_larger_better[METRIC_NAMES_DICT['higher_better_metrics']].merge(
        improvement_smaller_better[METRIC_NAMES_DICT['lower_better_metrics']], left_index=True, right_index=True)
    # reorganize the columns
    improvement = improvement[METRIC_NAMES_DICT['perf_metrics']+METRIC_NAMES_DICT['fairness_metrics']]
    return improvement


def improvement_bootstrap_iteration(i, df_baseline, df_corrected, ID_col, privileged_group, add_perf_difference):
    """Performs a single bootstrap iteration."""

    df_baseline_bootstrapped, df_corrected_bootstrapped = get_paired_bootstrap_dataframes(
        df_baseline, df_corrected,ID_col=ID_col)
    improvement = fairness_improvement(
        df_baseline_bootstrapped, df_corrected_bootstrapped, privileged_group=privileged_group,add_perf_difference=add_perf_difference)

    return improvement



def bootstrap_improvement_test(df_baseline, df_corrected, privileged_group, n_bootstrap=1000,add_perf_difference=True,ID_col='slide'):
    # raise NotImplementedError
    # dfs_improvement = []
    # for i in tqdm(range(n_bootstrap),miniters=n_bootstrap//10):
    #     df_baseline_bootstrapped, df_corrected_bootstrapped = get_paired_bootstrap_dataframes(
    #         df_baseline, df_corrected,ID_col=ID_col)
    #     improvement = fairness_improvement(
    #         df_baseline_bootstrapped, df_corrected_bootstrapped, privileged_group=privileged_group,add_perf_difference=add_perf_difference)
    #     dfs_improvement.append(improvement)
    # df_improvement_bootstrap = pd.concat(dfs_improvement)

    def parallelized_bootstrap(bootstrap_n, df_baseline, df_corrected, privileged_group, add_perf_difference):
        """Parallelizes the bootstrap process using multiprocessing.Pool."""

        bootstrap_results = []

        with Pool() as pool:
            # pool_func = partial(bias_bootstrap_iteration, df=df, df_g=df_g, privileged_group=privileged_group, add_perf_difference=add_perf_difference)
            pool_func = partial(improvement_bootstrap_iteration, df_baseline=df_baseline, df_corrected=df_corrected, ID_col=ID_col, privileged_group=privileged_group, add_perf_difference=add_perf_difference)
            bootstrap_results = list(tqdm(pool.imap(pool_func, range(bootstrap_n)), total=bootstrap_n, miniters=bootstrap_n//10))
            

        # df_bootstrap = pd.DataFrame.from_records(bootstrap_results)[METRIC_NAMES_DICT['fairness_metrics']]
        df_improvement_bootstrap = pd.concat(bootstrap_results)
        return df_improvement_bootstrap
    df_improvement_bootstrap = parallelized_bootstrap(n_bootstrap, df_baseline, df_corrected, privileged_group, add_perf_difference)
    # calculate actual improvement
    improvement = fairness_improvement(
        df_baseline, df_corrected, privileged_group=privileged_group,add_perf_difference=add_perf_difference)
    # calculate p-value
    df_p_better = improvement.copy()
    df_p_worse = improvement.copy()
    for i, row in improvement.iterrows():
        for col in improvement.columns:
            bootstrap_values = df_improvement_bootstrap[col].loc[i].dropna()
            p_better = (bootstrap_values >= row[col]).sum() / len(bootstrap_values)
            p_worse = (bootstrap_values <= row[col]).sum() / len(bootstrap_values)
            df_p_better.loc[i, col] = p_better
            df_p_worse.loc[i, col] = p_worse
    return improvement, df_p_better, df_p_worse
def fairness_improvement_groupLevel(df,add_perf_difference=True):
    METRIC_NAMES_DICT = get_metric_names(add_perf_difference=add_perf_difference)

    df_corrected = df.loc[df['model']=='corrected'].set_index(['fold'])
    df_baseline = df.loc[df['model']=='baseline'].set_index(['fold'])
    higher_better_metrics = METRIC_NAMES_DICT['higher_better_metrics']
    lower_better_metrics = METRIC_NAMES_DICT['lower_better_metrics']
    higher_better_metrics = [x if x not in METRIC_NAMES_DICT['perf_metrics'] else f'{x}_minority' for x in higher_better_metrics] # add minority to the performance metrics
    lower_better_metrics = [x if x not in METRIC_NAMES_DICT['perf_metrics'] else f'{x}_minority' for x in lower_better_metrics] # add minority to the performance metrics
    
    higher_better_metrics = list(set(df_baseline.columns).intersection(set(higher_better_metrics)))
    lower_better_metrics = list(set(df_baseline.columns).intersection(set(lower_better_metrics)))

    df_improv = df_baseline.copy()
    df_improv.drop(columns=['model'],inplace=True)
    df_improv[higher_better_metrics] = df_corrected[higher_better_metrics] - df_baseline[higher_better_metrics]
    df_improv[lower_better_metrics] = df_baseline[lower_better_metrics] - df_corrected[lower_better_metrics]
    df_mean_improv = df_improv.mean()
    # for smaller better metrics, the improvement is the difference between the baseline and corrected

    return df_improv, df_mean_improv
    
def bootstrap_improvement_test_groupLevel(dfs_baseline, dfs_corrected, privileged_group, n_bootstrap=1000,add_perf_difference=True):
    ## get the fairness for each group
    METRIC_NAMES_DICT = get_metric_names(add_perf_difference=add_perf_difference)

    baseline_metrics_list = []
    corrected_metrics_list = []
    for i, df in enumerate(dfs_baseline):
        fairResult = FairnessMetrics(
            df[f'pred'].to_numpy(),
            df[f'prob'].to_numpy(),
            df['label'].to_numpy(),
            df['sens_attr'].astype(str).to_numpy(), privileged_group=privileged_group,add_perf_difference=add_perf_difference)        
        fairResult = pd.DataFrame(fairResult)
        fairResult.insert(2, 'fold', i)

        baseline_metrics_list.append(fairResult)
    for i, df in enumerate(dfs_corrected):

        fairResult = FairnessMetrics(
            df[f'pred'].to_numpy(),
            df[f'prob'].to_numpy(),
            df['label'].to_numpy(),
            df['sens_attr'].astype(str).to_numpy(), privileged_group=privileged_group,add_perf_difference=add_perf_difference)        
        fairResult = pd.DataFrame(fairResult)
        fairResult.insert(2, 'fold', i)

        corrected_metrics_list.append(fairResult)
    df_baseline = pd.concat(baseline_metrics_list)
    df_corrected = pd.concat(corrected_metrics_list)
    df_baseline['model'] = 'baseline'
    df_corrected['model'] = 'corrected'
    ### only keep the minority group
    df_baseline = df_baseline.loc[df_baseline['group_type']=='minority'].reset_index(drop=True)
    df_corrected = df_corrected.loc[df_corrected['group_type']=='minority'].reset_index(drop=True)
    dropped_col = df_baseline[['sensitiveAttr','group_type']].iloc[0]
    df_baseline.drop(columns=['sensitiveAttr','group_type','N_0','N_1'],inplace=True)
    df_corrected.drop(columns=['sensitiveAttr','group_type','N_0','N_1'],inplace=True)
    
    for col in METRIC_NAMES_DICT['perf_metrics']:
        df_baseline[f'{col}_minority'] = df_baseline.pop(col)
        df_corrected[f'{col}_minority'] = df_corrected.pop(col)
        
    ###
    df = pd.concat([df_baseline, df_corrected]).reset_index(drop=True)
    ## calculate the improvement
    df_improv, df_mean_improv = fairness_improvement_groupLevel(df)
    df_mean_improv = pd.DataFrame(df_mean_improv).T
    ## bootstrap
    dfs_mean_improv_bootstrap = []
    for i in tqdm(range(n_bootstrap),miniters=n_bootstrap//10):
        # df_bootstrap = df.sample(frac=1,replace=True)
        df_bootstrap = df.groupby('fold').sample(frac=1,replace=True).reset_index()
        df_bootstrap[['fold','model']] = df[['fold','model']]
        df_improv_bootstrap, df_mean_improv_bootstrap = fairness_improvement_groupLevel(df_bootstrap)
        dfs_mean_improv_bootstrap.append(df_mean_improv_bootstrap)
    df_mean_improv_bootstrap = pd.concat(dfs_mean_improv_bootstrap,axis=1).T
    
    # calculate p-value
    df_p_better = df_mean_improv.copy()
    df_p_worse = df_mean_improv.copy()
    for col in df_mean_improv.columns:
        bootstrap_values = df_mean_improv_bootstrap[col].dropna()
        p_better = (bootstrap_values >= df_mean_improv[col].iloc[0]).sum() / len(bootstrap_values)
        p_worse = (bootstrap_values <= df_mean_improv[col].iloc[0]).sum() / len(bootstrap_values)
        df_p_better[col] = p_better
        df_p_worse[col] = p_worse
    for df in [df_mean_improv,df_p_better,df_p_worse]:
        df.insert(0,'sensitiveAttr',dropped_col['sensitiveAttr'])
        df.insert(1,'group_type',dropped_col['group_type'])
    return df_mean_improv, df_p_better, df_p_worse

def CV_bootstrap_improvement_test(
    dfs_baseline, dfs_corrected, privileged_group=None, n_bootstrap=1000,aggregate_method='fisher',add_perf_difference=True,
    ID_col='slide'):
    '''
    Estimate the improvement of the corrected model over the baseline model for all folds
    Input:
        dfs_baseline: list of pd.DataFrame, the baseline results
            Each pd.DataFrame contains the following columns:
                * sens_attr: sensitive attributes
                * prob: model probabilities
                * label: ground truth labels
                * pred:  model predictions
        dfs_corrected: list of pd.DataFrame, the corrected results
            Each pd.DataFrame contains the following columns:
                * sens_attr: sensitive attributes
                * prob: model probabilities
                * label: ground truth labels
                * pred:  model predictions
        n_bootstrap: int, number of bootstrap iterations
        aggregate_method: str, method to aggregate p-values. Options are:
            - 'concatenate': concatenate the input data
            - 'fisher', 'pearson', 'tippett', 'stouffer', 'mudholkar_george': methods to combine p-values, see scipy.stats.combine_pvalues for details
        privileged_group: str, the privileged group
        add_perf_difference: bool, whether to add performance difference as a fairness metric
        ID_col: str, the column name for the ID
        
    Output:
        df_p_better: pd.DataFrame, the p-values for significant improvement
        df_p_worse: pd.DataFrame, the p-values for significant worsening
        
    '''
    
    if aggregate_method == 'concatenate':
        # if the method is concatenate, we concatenate the data and return a single p-value
        df_baseline = pd.concat(dfs_baseline)
        df_corrected = pd.concat(dfs_corrected)
        df_improv, df_p_better, df_p_worse = \
            bootstrap_improvement_test(
            df_baseline, df_corrected, n_bootstrap=n_bootstrap, privileged_group=privileged_group,add_perf_difference=add_perf_difference,ID_col=ID_col)
        return df_improv, df_p_better, df_p_worse
    elif aggregate_method in ['fisher', 'pearson', 'tippett', 'stouffer', 'mudholkar_george']:
        # if the method is fisher or stouffer, we calculate the p-value for each fold
        dfs_improv = []
        dfs_p_better = []
        dfs_p_worse = []
        for i, (df_baseline, df_corrected) in enumerate(zip(dfs_baseline, dfs_corrected)):
            df_improv, df_p_better, df_p_worse = \
                bootstrap_improvement_test(
                df_baseline, df_corrected, n_bootstrap=n_bootstrap, privileged_group=privileged_group,add_perf_difference=add_perf_difference,ID_col=ID_col)
            df_improv.insert(0, 'fold', i)
            df_p_better.insert(0, 'fold', i)
            df_p_worse.insert(0, 'fold', i)
            dfs_improv.append(df_improv)
            dfs_p_better.append(df_p_better)
            dfs_p_worse.append(df_p_worse)
        ## concatenate the p-values

        df_p_better = pd.concat(dfs_p_better)
        df_p_worse = pd.concat(dfs_p_worse)
        df_improv = pd.concat(dfs_improv)
        ## aggregate the p-values
        df_p_combined = dfs_p_better[0].copy()
        for i, row in df_p_combined.iterrows():
            for col in df_p_combined.columns:
                pvals = df_p_better[col].loc[i]
                meta_res = combine_pvalues(pvals, method=aggregate_method, weights=None)
                df_p_combined[col].loc[i] = meta_res.pvalue
        df_p_combined['fold'] = f'{aggregate_method}_combined'
        df_p_better = pd.concat([df_p_better, df_p_combined])
        
        df_p_combined = dfs_p_worse[0].copy()
        for i, row in df_p_combined.iterrows():
            for col in df_p_combined.columns:
                pvals = df_p_worse[col].loc[i]
                meta_res = combine_pvalues(pvals, method=aggregate_method, weights=None)
                df_p_combined[col].loc[i] = meta_res.pvalue
        df_p_combined['fold'] = f'{aggregate_method}_combined'
        df_p_worse = pd.concat([df_p_worse, df_p_combined])
        return df_improv, df_p_better, df_p_worse
    elif aggregate_method == 'groupwise':
        # if the method is groupwise, we estimate the fairness metrics first, and then perform bootstraping on population level
        df_improv, df_p_better, df_p_worse = bootstrap_improvement_test_groupLevel(
            dfs_baseline, dfs_corrected, n_bootstrap=n_bootstrap, privileged_group=privileged_group,add_perf_difference=add_perf_difference)
        return df_improv, df_p_better, df_p_worse

def get_paired_bootstrap_dataframes(df_baseline, df_corrected, ID_col='slide'):
    '''
    get the paired bootstrap dataframe for pair-sampled test
    NOTE: 
        This is different from utils.get_bootstrap_stats, which is for independent-sampled test.
        This bootstrap only within the same ID.
    input:
        df_baseline: pd.DataFrame, the baseline results
        df_corrected: pd.DataFrame, the corrected results
    output:
        df_baseline_bootstrapped: pd.DataFrame, the bootstrapped baseline results
        df_corrected_bootstrapped: pd.DataFrame, the bootstrapped corrected results
    '''
    # get the paired bootstrap dataframe
    df = pd.concat([df_baseline, df_corrected], axis=0)
    df_baseline_bootstrapped = df.groupby(
        ID_col).sample(n=1).reset_index(drop=True)
    df_corrected_bootstrapped = df.groupby(
        ID_col).sample(n=1).reset_index(drop=True)
    return df_baseline_bootstrapped, df_corrected_bootstrapped

