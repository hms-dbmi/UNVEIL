import os
from os.path import join, basename, dirname
import argparse
import torch
import wandb
import glob
from torch.utils.data import DataLoader
from tqdm import tqdm
from network import ClfNet
from util import *
from fairness_utils import *
import numpy as np
from pathlib import Path

from dataset import *
from framework import *
import datetime
from sklearn.metrics import roc_auc_score
import copy
import traceback
import warnings

def list_available_genes(args,use_abbr=False):
    if args.data_source == 'TCGA':
        '''
        for TCGA, the gene list is obtained from the TCGA pan-cancer atlas 2018 dataset spreadsheets
        '''
        directory_path = f"./tcga_pan_cancer/{args.cancer[0].lower()}_tcga_pan_can_atlas_2018"
        if os.path.isdir(directory_path):
            gene_list_all = []
            gene_dict = {}
            for types in os.listdir(directory_path):
                if types == 'Common Genes':
                    geneType = 'Common Genes'
                elif types == 'Targeted Drugs for Genes':
                    geneType = 'Mutated Genes'
                if os.path.isdir(join(directory_path, types)):
                    genes = []
                    for gName in os.listdir(join(directory_path, types)):
                        # full name of the folder
                        geneName = "_".join(gName.split('_')[1:-1])
                        geneName_short = geneName.split('-')[0]  # the actual gene name
                        genes.append(geneName_short)
                        gene_list_all.append(geneName_short if use_abbr else geneName)
                    gene_dict[geneType] = genes
            # print
            print("Available genes:")
            for geneType, geneList in gene_dict.items():
                print(f"\t{geneType}:\t{', '.join(geneList)}")
            # remove duplicates
            gene_list_all = list(set(gene_list_all))
            return gene_list_all
        else:
            raise FileNotFoundError(f"Directory not found: {directory_path}")
    else:
        '''
        For other data sources, the gene list is obtained directly from the clinica data files
        '''
        ## list the genes available in the clinica data files
        ## load dataset configuration
        with open(args.dataset_config_yaml, 'r') as stream:
            try:
                dataset_configs = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                traceback.print_exc()
                print(exc)
        dataset_config = dataset_configs[args.data_source]
        mutation_kw = dataset_config['args']['mutation_kw']
        strClinicalInformationPath = dataset_config['args']['strClinicalInformationPath']
        clinical_file = join(strClinicalInformationPath,f'{args.cancer[0].upper()}_clinical_information.csv')
        assert os.path.exists(clinical_file), f"File not found: {clinical_file}"
        clinical_df = pd.read_csv(clinical_file)
        columns = clinical_df.columns
        gene_names = [col for col in columns if col.endswith(mutation_kw)]
        gene_names_short = [col.split(mutation_kw)[0] for col in gene_names]
        return gene_names_short if use_abbr else gene_names


def main_train_valid(args):
    directory_path = f"./tcga_pan_cancer/{args.cancer[0]}_tcga_pan_can_atlas_2018"
    if args.data_source != 'TCGA':
        raise notImplementedError("Only TCGA data source is supported for now. If you want to train on other datasets you need to implement this.")

    if os.path.isdir(directory_path):
        for types in os.listdir(directory_path):
            if types == 'Common Genes':
                geneType = 'Common Genes'
            elif types == 'Targeted Drugs for Genes':
                geneType = 'Mutated Genes'

            if os.path.isdir(f'{directory_path}/{types}/'):
                for gName in os.listdir(f"{directory_path}/{types}/"):
                    try:
                        if gName.endswith('_'):
                            gName = gName[:-1]
                        geneName = "_".join(gName.split('_')[1:])
                        geneName_short = geneName.split('-')[0]
                        
                        if args.gene not in geneName:
                            continue
                        ##
                        print(f"Gene: {geneName_short}, fold: {args.curr_fold}")
                        cancer_folder = f"{str(args.task)}_{args.cancer[0]}_{geneType}_{geneName}_{args.partition}"
                        dir = join(
                            args.model_path, args.cancer[0], f"{cancer_folder}{reweight_str}/")
                        if not os.path.exists(dir):
                            os.makedirs(dir, exist_ok=True)
                            print(f"directory: {dir} created")
                        
                        # check_results_path = join(dir, f"inference_results.csv")
                        # if os.path.exists(check_results_path) and args.skip_existing:
                        #     print(f"Results exist: {check_results_path}. Skip training.")
                        #     continue
                        
                        ##
                        ## new function to check if the task should be skipped
                        if should_skip_task(args, dir):
                            continue

                        if not args.pretrained:
                            # If the model is not pretrained, specify the run ID based on the existing runs
                            
                            print("Fetching latest run IDs...")
                            folder_names = os.listdir(dir)
                            subfolders = [folder for folder in folder_names if os.path.isdir(
                                os.path.join(dir, folder))]
                            if not subfolders:
                                max_index = 1
                            else:
                                model_indexes = [int(name.split('_')[0])
                                                for name in subfolders]
                                max_index = max(model_indexes)
                                if args.partition == 1:
                                    max_index += 1
                                if args.partition in [2,'fixedKFold'] and args.curr_fold == 0:
                                    max_index += 1
                        else:
                        ## If the model requires finetuning get the maximum run ID from baseline models
                            print("Fetching latest run IDs from baseline...")
                            baseline_args = copy.deepcopy(args)
                            baseline_args.train_method = "baseline"
                            # Set baseline defaults
                            baseline_args.reweight = False
                            baseline_args.selection = 'AUROC'
                            baseline_args.pretrained = False
                            baseline_args.finetune_layer_names = None
                            # baseline_max_index, baseline_results_folders = test_os_settings(baseline_args)
                            baseline_max_index, baseline_results_folders = None, None
                            for baseline_max_index, baseline_results_folders in test_os_settings(baseline_args):
                                break
                            if baseline_max_index is None:
                                print("Baseline results not found. Skip training.")
                                continue
                            max_index = baseline_max_index
                            print(f"Max run index from baseline: {max_index}")
                        print(f"Assigning run ID: {max_index}")

                        job, task = wandb_setup(args, max_index, gene=geneName_short)

                        # Dataset preparation
                        args.geneType = geneType
                        args.geneName = geneName
                        data = generate_dataset(args)


                        df = data.train_valid_test(args.split_ratio)

                        num_classes = len(df["label"].unique())

                        # Initialize demographic-informed agent (without save_dir for now)
                        demographic_agent = None
                        if args.use_demographic_agent:
                            from pathlib import Path
                            
                            # Select agent class based on strategy
                            if args.demographic_strategy == 'unified':
                                from unified_demographic_agent import UnifiedDemographicAgent as DemographicPatchAgent
                            else:
                                from demographic_agent import DemographicPatchAgent
                            
                            try:
                                # Pass all agent parameters
                                agent_kwargs = {
                                    'attribute': args.demographic_attribute,
                                    'cancer': args.cancer[0],
                                    'foundation_model': args.foundation_model,
                                    'gene': geneName_short,
                                    'strategy': args.demographic_strategy,
                                    'base_filter_percentile': args.demographic_base_percentile,
                                    'adaptive_filtering': args.demographic_adaptive,
                                    'use_correctness_weighting': args.demographic_use_correctness_weighting,
                                    'save_dir': None  # Will be set later after model_save_path is created
                                }
                                
                                # Add multi-factor routing parameter if using unified strategy
                                if args.demographic_strategy == 'unified':
                                    if hasattr(args, 'demographic_use_v6_routing'):
                                        agent_kwargs['use_v6_routing'] = args.demographic_use_v6_routing
                                
                                demographic_agent = DemographicPatchAgent(**agent_kwargs)
                                
                                print(f"[Main] Demographic signal-aware agentic scheduling initialized for {args.demographic_attribute}-{args.cancer[0]}-{geneName_short}")
                                if hasattr(args, 'demographic_use_v6_routing') and args.demographic_use_v6_routing:
                                    print(f"[Main] Using multi-factor routing strategy")
                            except Exception as e:
                                print(f"[Main] Warning: Could not initialize agent: {e}")
                                print(f"[Main] Continuing without agent")
                                demographic_agent = None

                        if args.partition == 1:
                            train_ds, val_ds, test_ds = get_datasets(
                                df, args.task, "vanilla", None, feature_type=args.feature_type,reweight_method=args.reweight_method, reweight_cols=args.reweight_cols, max_train_tiles=args.max_train_tiles, demographic_agent=demographic_agent)
                        elif args.partition in [2, "fixedKFold"]:
                            train_ds, val_ds, test_ds = get_datasets(
                                df, args.task, "kfold", args.curr_fold, feature_type=args.feature_type, reweight_method=args.reweight_method, reweight_cols=args.reweight_cols, max_train_tiles=args.max_train_tiles, demographic_agent=demographic_agent)
                        task_collate_fn = task_collate_fn_settings(args)
                        train_dl = DataLoader(train_ds, collate_fn=task_collate_fn, sampler=train_ds.sampler,
                                                batch_size=args.batch_size,  pin_memory=False, num_workers=args.n_workers)
                        val_dl = DataLoader(val_ds, collate_fn=task_collate_fn, batch_size=args.eval_batch_size,
                                            shuffle=False,  pin_memory=False, num_workers=args.n_workers)
                        test_dl = DataLoader(test_ds, collate_fn=task_collate_fn, batch_size=args.eval_batch_size,
                                                shuffle=False,  pin_memory=False, num_workers=args.n_workers)

                        # initialize torch
                        torch.manual_seed(args.seed)
                        if not args.pretrained:
                            model = get_model(args,num_classes)
                        if args.pretrained:
                            model = load_pretrained_weights(
                                args, num_classes, max_index)

                        model = model.to(args.device)

                        # Settings
                        optimizer = optimizer_settings(args, model)

                        scheduler = torch.optim.lr_scheduler.StepLR(
                            optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
                        
                        loss_fn = loss_fn_settings(args, train_ds)

                        # Output folder
                        model_save_path = join(dir, job)
                        os.makedirs(model_save_path, exist_ok=True)

                        # Set agent save directory now that model_save_path is created
                        if demographic_agent is not None:
                            from pathlib import Path
                            agent_save_dir = Path(model_save_path) / "agent_data"
                            demographic_agent.save_dir = agent_save_dir
                            os.makedirs(agent_save_dir, exist_ok=True)
                            print(f"[Main] Agent data will be saved to: {agent_save_dir}")

                        # save args
                        save_args(args, model_save_path, filename='args.json')

                        epochs_without_improvement = 0
                        epoch_record = 0
                        best_performance = 0.
                        best_fairness = 9999.
                        best_es_metric = 0.
                        group_samples = {}

                        # Training/evaluation process
                        for epoch in range(args.epochs):
                            model.train()
                            avg_train_loss, avg_train_fair_loss, avg_group_loss, avg_overall_loss = run(
                                args, train_dl, model, num_classes, "yellow", loss_fn, optimizer, epoch)
                            scheduler.step()

                            model.eval()
                            with torch.no_grad():
                                eval_results, avg_eval_loss, avg_eval_fair_loss, avg_eval_group_loss, avg_eval_overall_loss = run(args, val_dl, model, num_classes,
                                                                                                                                    "blue", loss_fn, optimizer, epoch)
                                labels, senAttrs, events, true_survival_times, predicted_survival_times, predictions, probs, logits, caseIds, slideIds = eval_results
                                senAttrs = [data.dictInformation['sensitive'][x] for x in senAttrs]

                            if num_classes > 2:
                                raise NotImplementedError("Not refacroting this part for now")
                                results = FairnessMetricsMultiClass(
                                    predictions, labels, senAttrs)
                                acc = results["OverAllAcc"]
                                fairness = results["EOdd"]
                                criterion = 0                   # 0: performance 1: fairness 2: both
                                if args.selection == "EOdd":
                                    fairness = results["EOdd"]
                                    criterion = 1
                                elif args.selection == "avgEOpp":
                                    fairness = (
                                        results["EOpp0"] + results["EOpp1"]) / 2
                                    criterion = 1
                                elif args.selection == "OverAllAcc":
                                    criterion = 0

                                if criterion == 0:
                                    if acc > best_performance:
                                        best_performance = acc
                                        torch.save(model.state_dict(), Path(
                                            model_save_path) / "model.pt")
                                        epoch_record = epoch
                                        epochs_without_improvement = 0
                                        print(
                                            f"Epoch:{epoch_record}, {args.selection}:{best_performance}")
                                elif criterion == 1:
                                    if fairness > 0 and fairness < best_fairness:
                                        best_fairness = fairness
                                        torch.save(model.state_dict(), Path(
                                            model_save_path) / "model.pt")
                                        # save_thresholds(labels, probs,senAttrs, model_save_path)  ## save thresholds
                                        epoch_record = epoch
                                        epochs_without_improvement = 0
                                        print(
                                            f"Epoch:{epoch_record}, {args.selection}:{best_fairness}")

                                temp = {"Avg_Loss(train)": avg_train_loss,
                                        "Avg_Loss(valid)": avg_eval_loss,
                                        "Group(M) Majority": results["TOTALACC"][1],
                                        "Group(m) Minority": results["TOTALACC"][0],
                                        "Fair_Loss(train)": avg_train_fair_loss,
                                        "Fair_Loss(valid)": avg_eval_fair_loss,
                                        "Group_Loss(train)": avg_group_loss,
                                        "Group_Loss(valid)": avg_eval_group_loss,
                                        "Overall Loss(train)": avg_overall_loss,
                                        "Overall Loss(valid)": avg_eval_overall_loss,
                                        }
                                wandb_record = {**temp, **results}
                                wandb.log(wandb_record)

                            elif num_classes == 2 and args.task != 3:
                                probs = probs[:, 1]
                                ##do not print the warnings 
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore")
                                    auroc = roc_auc_score_nan(labels, probs)
                                    threshold = Find_Optimal_Cutoff(
                                        labels, probs, senAttrs, method=args.cutoff_method)
                                    predictions = torch.ge(torch.tensor(
                                        predictions), threshold).int()
                                    results = FairnessMetrics(
                                        predictions, probs, labels, senAttrs)
                                ## Log the results
                                temp = {
                                    "AUROC": auroc,
                                    "Threshold": threshold,
                                    "Avg_Loss(train)": avg_train_loss,
                                    "Avg_Loss(valid)": avg_eval_loss,
                                    "Fair_Loss(train)": avg_train_fair_loss,
                                    "Fair_Loss(valid)": avg_eval_fair_loss,
                                    "Group_Loss(train)": avg_group_loss,
                                    "Group_Loss(valid)": avg_eval_group_loss,
                                    "Overall Loss(train)": avg_overall_loss,
                                    "Overall Loss(valid)": avg_eval_overall_loss,
                                }
                                wandb_record = {**temp, **results}
                                wandb.log(wandb_record)
                                
                                ## early stopping
                                if  args.selection is None:
                                    
                                    torch.save(model.state_dict(), Path(
                                        model_save_path) / "model.pt")
                                    ## suppress warnings
                                    with warnings.catch_warnings():
                                        warnings.simplefilter("ignore")
                                        # save thresholds
                                        save_thresholds(args, labels, probs, senAttrs, model_save_path,
                                                        save_metrics=True,
                                                        save_metrics_postfix=f'valid(optimal_{args.cutoff_method})',
                                                        log_method=args.cutoff_method)
                                    continue
                                
                                
                                ES_OPT_MAPPING = {
                                    'EOdd': 'min',
                                    'avgEOpp': 'min',
                                    'AUROC': 'max',
                                    'GroupMacroAUROC': 'max',
                                    'GroupMinAUROC': 'max',
                                    'GroupMacroRecall': 'max',
                                    'GroupMinRecall': 'max',
                                    'loss': 'min',
                                }
                                ES_METRIC_MAPPING = {
                                    'EOdd': results["EOddMax"],
                                    'avgEOpp': (results["EOpp0"] + results["EOpp1"]) / 2,
                                    'AUROC': auroc,
                                    'GroupMacroAUROC': results["AUC"].mean(),
                                    'GroupMinAUROC': results["AUC"].min(),
                                    'GroupMacroRecall': np.mean(np.concatenate([results["TPR"], results["TNR"]])),
                                    'GroupMinRecall': np.min(np.concatenate([results["TPR"], results["TNR"]])),
                                    'loss':  avg_eval_overall_loss,
                                }
                                criterion = ES_OPT_MAPPING[args.selection]
                                ES_metric = ES_METRIC_MAPPING[args.selection]
                                

                                # criterion = 0                   # 0: performance 1: fairness 2: both
                                # fairness = results["EOdd"]
                                # if args.selection == "EOdd":
                                #     fairness = results["EOdd"]
                                #     criterion = 1
                                # elif args.selection == "avgEOpp":
                                #     fairness = (
                                #         results["EOpp0"] + results["EOpp1"]) / 2
                                #     criterion = 1
                                # elif args.selection == "AUROC":
                                #     criterion = 0

                                if epoch == 0:
                                    best_es_metric = ES_metric
                                    torch.save(model.state_dict(), Path(
                                        model_save_path) / "model.pt")
                                    
                                    with warnings.catch_warnings():
                                        warnings.simplefilter("ignore")
                                        save_thresholds(args, labels, probs, senAttrs, model_save_path,
                                                        save_metrics=True,
                                                        save_metrics_postfix=f'valid(optimal_{args.cutoff_method})',
                                                        log_method=args.cutoff_method)
                                if (criterion == 'max' and ES_metric > best_es_metric) or (criterion == 'min' and ES_metric < best_es_metric):
                                    best_es_metric = ES_metric
                                    torch.save(model.state_dict(), Path(
                                        model_save_path) / "model.pt")
                                    with warnings.catch_warnings():
                                        warnings.simplefilter("ignore")
                                        save_thresholds(args, labels, probs, senAttrs, model_save_path,
                                                        save_metrics=True,
                                                        save_metrics_postfix=f'valid(optimal_{args.cutoff_method})',
                                                        log_method=args.cutoff_method)
                                    epoch_record = epoch
                                    
                                    epochs_without_improvement = 0
                                    print(
                                        f"Epoch:{epoch_record}, {args.selection}:{best_es_metric}")
                                elif args.patience is not None:  # if early stopping is enabled
                                    epochs_without_improvement += 1

                            if args.patience is not None:
                                if epochs_without_improvement >= args.patience:
                                    print(
                                        f"Early stopping at epoch {epoch} after {args.patience} epochs without improvement")
                                    break
                        wandb.finish()
                        
                        # Save agent logs if agent was used
                        if demographic_agent is not None:
                            demographic_agent.save_logs()

                        print(
                            f"Epoch:{epoch_record}, {args.selection}:{best_es_metric}")
                    except Exception as e:
                        print("Error in training: ", str(e))
                        print("Skip gene: ", geneName_short)

                        traceback.print_exc()


def main_test(args,inference_mode:Literal['valid', 'test','train','all']='test'):
    # Dataset preparation
    results_path = join(args.model_path, args.cancer[0].lower())
    inference_results_path = join(args.inference_output_path, args.cancer[0].lower())
    inf_mode_prefix_map = {'valid': 'valid_', 'test': '', 'train': 'train_', 'all': 'all_'}
    inf_mode_prefix = inf_mode_prefix_map[inference_mode]
    
    # iterate through the trained models
    for models in os.listdir(results_path):
        # the folders should be in the format of task_cancer_geneType_geneName_freq_partition
        # e.g. 4_brca_Common Genes_CDH1-Percentage_12.2_2
        if len(models.split("_")) == 6:

            geneType = models.split("_")[2]
            geneName = "_".join(models.split("_")[3:-1])
            geneName_short = geneName.split('-')[0]
    
            if args.gene not in geneName:
                continue
            print(f"Gene: {geneName_short}")

            try:                
                args.geneType = geneType
                args.geneName = geneName if args.data_source == 'TCGA' else geneName_short
                data = generate_dataset(args,inference_mode)
                try:
                    df = data.train_valid_test()
                except Exception as e:
                    traceback.print_exc()
                    print("Error in generating dataset: ", str(e))
                    print("Skip gene: ", geneName_short)
                    
                    continue

                num_classes = len(df["label"].unique())

                auroc = 0.

                if args.partition == 1:
                    split_type = "vanilla"
                    folds = range(1)

                elif args.partition in [2,'fixedKFold']:
                    split_type = "kfold"
                    folds = range(N_FOLDS)

                    
                caseIds = [[] for _ in folds]
                logits = [[] for _ in folds]
                probs = [[] for _ in folds]
                predictions = [[] for _ in folds]
                labels = [[] for _ in folds]
                events = [[] for _ in folds]
                senAttrs = [[] for _ in folds]
                predicted_survival_times = [[] for _ in folds]
                true_survival_times = [[] for _ in folds]
                stages = [[] for _ in folds]
                features = [[] for _ in folds]
                for curr_fold in folds:
                    task_collate_fn = task_collate_fn_settings(args)
                    if inference_mode == 'all':
                        inference_ds = CancerDataset(df, args.task, split_type=split_type,feature_type=args.feature_type)
                    else:
                        train_ds, val_ds, test_ds = get_datasets(df, args.task, split_type, curr_fold,feature_type=args.feature_type)
                        inference_ds = val_ds if inference_mode == 'valid' else test_ds if inference_mode == 'test' else train_ds

                    inference_dl = DataLoader(inference_ds, batch_size=args.eval_batch_size, collate_fn=task_collate_fn,
                                            shuffle=False, pin_memory=False, num_workers=args.n_workers)
                    cancer_folder = f"{args.task}_{ '_'.join(args.cancer)}_{geneType}_{geneName}_{args.partition}"
                    gene_reweight_folder = case_insensitive_glob(join(results_path, f"{cancer_folder}{reweight_str}"))[0]
                    gene_weight_folder = case_insensitive_glob(join(results_path,  cancer_folder))[0]
                    model_names = os.listdir(gene_weight_folder)
                    subfolders = [folder for folder in model_names if os.path.isdir(
                        os.path.join(gene_weight_folder, folder))]
                    model_indexes = [int(name.split('_')[0])
                                        for name in subfolders]
                    max_index = max(model_indexes)

                    if args.pretrained:
                        model = get_model(args,num_classes)
                        reweight_names = os.listdir(gene_reweight_folder)
                        subfolders = [folder for folder in reweight_names if os.path.isdir(
                            join(gene_reweight_folder, folder))]
                        reweight_indexes = [
                            int(name.split('_')[0]) for name in subfolders]
                        if len(reweight_indexes) == 0:
                            print(
                                f"No reweighting results found for {cancer_folder}. Skip.")
                            continue
                        if args.weight_path == "":
                            max_reweight_index = max(reweight_indexes)
                        else:
                            max_reweight_index = int(args.weight_path)
                        weight_path_sstr = join(gene_reweight_folder,f"{max_reweight_index}_*_{curr_fold}","model.pt")
                        weight_paths = case_insensitive_glob(weight_path_sstr)
                        assert len(
                            weight_paths) > 0, f"reweight path not found: {weight_path_sstr}"
                        weight_path = weight_paths[0]

                        model.load_state_dict(torch.load(
                            weight_path, map_location=args.device), strict=False)
                        inference_results_path = Path(
                            weight_path).parent / f"{inf_mode_prefix}inference_results_fold{curr_fold}.csv"
                        result_path = Path(
                            weight_path).parent.parent / f"{max_reweight_index}_result.csv"

                    elif not args.pretrained:
                        model = get_model(args,num_classes)
                        weight_path_sstr = join(
                            results_path, f"{cancer_folder}/{max_index}_*_{curr_fold}/model.pt")
                        weight_paths = case_insensitive_glob(weight_path_sstr)
                        
                        assert len(
                            weight_paths) > 0, f"weight path not found: {weight_path_sstr}"
                        weight_path = weight_paths[0]
                        model.load_state_dict(torch.load(
                            weight_path, map_location=args.device), strict=False)
                        inference_results_path = Path(
                            weight_path).parent / f"{inf_mode_prefix}inference_results_fold{curr_fold}.csv"
                        result_path = Path(
                            weight_path).parent.parent / f"{max_index}_result.csv"
                    
                    if not "weight_path" in locals():
                        print(f"Weight path not found for {cancer_folder}. Skip.")
                        continue
                    if should_skip_task(args,  Path(weight_path).parent.parent, mode='inference',inference_mode=inference_mode):
                        continue
                    ## replace the model_path with the inference_output_path
                    inference_results_path = str(inference_results_path).replace(str(Path(args.model_path)), args.inference_output_path)
                    result_path = str(result_path).replace(str(Path(args.model_path)), args.inference_output_path)
                    os.makedirs(dirname(inference_results_path), exist_ok=True)
                    os.makedirs(dirname(result_path), exist_ok=True)


                    model.eval().to(args.device)

                    test_pbar = tqdm(enumerate(inference_dl), colour="blue", total=len(
                        inference_dl), mininterval=10)

                    with torch.no_grad():
                        for _, batch in test_pbar:
                            if args.task == 1 or args.task == 2 or args.task == 4:
                                wsi_embeddings, lengths, sensitive, label, group, case_id, folder_id = batch
                                ## inference with generic output format (tuple)
                                results = model(wsi_embeddings.to(
                                    args.device), sensitive.to(args.device),
                                    lengths=lengths,
                                    return_features=args.inference_output_features)
                                ##### parse the tuple
                                if args.inference_output_features:
                                    test_cancer_pred, feature = results
                                    features[curr_fold].append(feature.detach().cpu().numpy())
                                else:
                                    test_cancer_pred = results
                                #### add results to list
                                logits[curr_fold].append(
                                    test_cancer_pred.detach().cpu().numpy()[:, 1])
                                probs[curr_fold].append(torch.sigmoid(
                                    test_cancer_pred).detach().cpu().numpy()[:, 1])
                                predictions[curr_fold].append(torch.argmax(
                                    test_cancer_pred.detach().cpu(), dim=1).numpy())
                                labels[curr_fold].append(
                                    label.detach().cpu().numpy())
                                senAttrs[curr_fold].append(
                                    sensitive.detach().cpu().numpy())
                                caseIds[curr_fold].append(case_id)

                    # concatenate results from this fold
                    logits[curr_fold] = np.concatenate(logits[curr_fold])
                    probs[curr_fold] = np.concatenate(probs[curr_fold])
                    # predictions[curr_fold] = np.concatenate(predictions[curr_fold])
                    labels[curr_fold] = np.concatenate(labels[curr_fold])
                    if args.inference_output_features: 
                        features[curr_fold] = np.concatenate(features[curr_fold], axis=0)
                    senAttrs[curr_fold] = np.concatenate(
                        senAttrs[curr_fold])
                            
                    ## map the encoded sensitive attribute back to the original sensitive attribute
                    senAttrs[curr_fold]  = [data.dictInformation['sensitive'][x] for x in senAttrs[curr_fold] ]
                    # caseIds[curr_fold] = np.concatenate(caseIds[curr_fold])
                    # estimate the prediction
                    inpath = dirname(weight_path)
                    predictions[curr_fold] = get_predictions(
                        probs[curr_fold], inpath, method=args.cutoff_method)
                    caseIds[curr_fold] = [
                        item for sublist in caseIds[curr_fold] for item in sublist]

                    inference_results = pd.DataFrame({
                        "logits": logits[curr_fold],
                        "prob": probs[curr_fold],
                        "pred": predictions[curr_fold],
                        "label": labels[curr_fold],
                        "sens_attr": senAttrs[curr_fold],
                        "patient_id": caseIds[curr_fold],
                    })
                    save_metrics_summary(
                        args,labels[curr_fold],probs[curr_fold],
                        senAttrs[curr_fold], str(Path(weight_path).parent),
                        postfix=f'{inference_mode}(optimal_{args.cutoff_method})')

                    ### if output features is enabled, save the features and all metadata in npy format
                    if args.inference_output_features:
                        out_npy_path = str(Path(inference_results_path).parent / f"{inf_mode_prefix}inference_features_fold{curr_fold}.npy")
                        np.save(out_npy_path, {
                            "features": features[curr_fold],
                            "labels": labels[curr_fold],
                            "sens_attr": senAttrs[curr_fold],
                            "patient_id": caseIds[curr_fold],
                        })
                        print(f"Save features to: {out_npy_path}")

                    inference_results.to_csv(inference_results_path)
                    fold_auroc = roc_auc_score_nan(
                        labels[curr_fold], probs[curr_fold])
                    print(f"Fold {curr_fold} AUROC: {fold_auroc}")
                # concatenate results from all folds
            
                if not "weight_path" in locals():
                    print(f"Weight path not found for {cancer_folder}. Skip.")
                    continue
                if should_skip_task(args,  Path(weight_path).parent.parent, mode='inference',inference_mode=inference_mode):
                    continue
                
                logits = np.concatenate(logits)
                probs = np.concatenate(probs)
                predictions = np.concatenate(predictions)
                labels = np.concatenate(labels)
                senAttrs = np.concatenate(senAttrs)
                caseIds = [item for sublist in caseIds for item in sublist]
                # save results from all folds
                
                inference_results_path = Path(
                    weight_path).parent.parent / f"{inf_mode_prefix}inference_results.csv"
                
                ## Skip if the results already exist
                # check_results_path = Path(
                #     weight_path).parent.parent / f"{inf_mode_prefix}inference_results.csv"
                # if os.path.exists(check_results_path) and args.skip_existing:
                #     print(f"Results exist: {check_results_path}. Skip.")
                #     continue
                

                ##
                inference_results_path = str(inference_results_path).replace(str(Path(args.model_path)), args.inference_output_path)
                os.makedirs(dirname(inference_results_path), exist_ok=True)
                inference_results = pd.DataFrame({
                    "logits": logits,
                    "prob": probs,
                    "pred": predictions,
                    "label": labels,
                    "sens_attr": senAttrs,
                    "patient_id": caseIds,
                })
                inference_results.to_csv(inference_results_path)

                if num_classes > 2:
                    results = FairnessMetricsMultiClass(
                        predictions, labels, senAttrs)
                    pd.DataFrame(results).T.to_csv(result_path)
                    print(f"Save results to:{result_path}")

                elif num_classes == 2:
                    # probs = probs[:, 1]
                    auroc = roc_auc_score_nan(labels, probs)

                    # threshold = Find_Optimal_Cutoff(labels,probs,senAttrs,method=args.cutoff_method)
                    # predictions = np.where(probs > threshold, 1, 0)
                    results = FairnessMetrics(
                        predictions, probs, labels, senAttrs)
                    temp = {"AUROC": auroc}
                    results = {**temp, **results}
                    pd.DataFrame(results).T.to_csv(result_path)
                    print(f"Save results to:{result_path}")

                # fmtc = Metrics(predictions = predictions, probs=probs,labels = labels, sensitives = senAttrs, projectName = "proposed", verbose = True)
                # markdown = fmtc.getResults(markdownFormat=True)
                # if auroc != 0: markdown += f"{auroc:.4f}|"
                # print(markdown)
                print(pd.DataFrame(results).T)
                del model, results
            except Exception as e:
            # if False:
                traceback.print_exc()
                print("Error: ", str(e))

def main_CV(args):
    if args.partition == 1: # A single train-validation-test split
        main_train_valid(args)
        # main_test(args)
        for mode in args.inference_mode:
            main_test(args, inference_mode=mode)
    elif args.partition in [2,'fixedKFold']:
        if args.curr_fold is None: # K-fold cross validation, run all folds
            for curr_fold in range(N_FOLDS):
                print(f"Gene: {args.gene}, Fold: {curr_fold}")
                fold_args = copy.deepcopy(args)
                fold_args.curr_fold = curr_fold
                main_train_valid(fold_args)
            # main_test(args)
            for mode in args.inference_mode:
                main_test(args, inference_mode=mode)
            run_significance_test(args,inference_mode="test")
        else:
            all_folds = deepcopy(args.curr_fold)
            for curr_fold in all_folds:
                print(f"Gene: {args.gene}, Fold: {curr_fold}")
                fold_args = copy.deepcopy(args)
                fold_args.curr_fold = curr_fold
                main_train_valid(fold_args)
                if curr_fold == N_FOLDS-1:
                    print("starting testing")
                    # main_test(args)
                    for mode in args.inference_mode:
                        main_test(args, inference_mode=mode)
                    run_significance_test(args,inference_mode="test")
        
    else: # K-fold cross validation, run specific fold
        main_train_valid(args)
        if  args.curr_fold == N_FOLDS-1:
            print("starting testing")
            # main_test(args)
            for mode in args.inference_mode:
                main_test(args, inference_mode=mode)
            run_significance_test(args,inference_mode="test")

if __name__ == "__main__":
    print("Current date and time: ", datetime.datetime.now())
    args = parse_args()
    print(args)
    gene_list = list_available_genes(args,use_abbr=False if args.data_source == 'TCGA' else True)
    for gene in gene_list:
        if args.genes is not None:
            if not any([s in gene for s in args.genes]):
                continue
        args.gene = gene
        if args.sig_test_only:
            print(f"Gene: {args.gene}")
            # for mode in args.inference_mode:
            run_significance_test(args,inference_mode="test")
        elif args.inference_only:
            for mode in args.inference_mode:
                main_test(args, inference_mode=mode)
            run_significance_test(args,inference_mode="test")
        else:
            main_CV(args)
