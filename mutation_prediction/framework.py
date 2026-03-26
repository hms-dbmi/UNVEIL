import argparse
import os
import json
import wandb
import torch
import glob
from pathlib import Path
from tqdm import tqdm
from network import ClfNet, MLP, ClfNetwSensitive, MLPwSensitive
from util import *
from fairness_utils import *
from fairmetric import *
from os.path import join, dirname,basename
import typing
from typing import List, Union, Dict
from dataset import *
from sklearn.metrics import roc_auc_score
import importlib
import traceback
from bootstrap_tests import CV_bootstrap_bias_test, CV_bootstrap_improvement_test
import yaml
from typing import Literal, Union

# Cross-validation configuration
N_FOLDS = 4


FOUNDATION_MDL_FEATURE_TYPE_MAP = {
    'CHIEF': 'tile',
    'UNI': 'tile',
    'GIGAPATH': 'tile',
    'VIRCHOW2': 'tile',
    'TITAN': 'slide',
    'GIGAPATH_WSI':'slide',
    'CHIEF_WSI':'slide',
}

def int_or_str(value):
    """
    A custom type function for argparse that attempts to convert a value to an
    integer, and if it fails, returns it as a string.
    """
    try:
        return int(value)
    except ValueError:
        return value

def parse_args(input_args=None, print_args=False):

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cancer",
        nargs='+',
        default=None,
        required=True,
        help="Cancers are the targets for this task.",
    )
    parser.add_argument(
        "--fair_attr",
        default=None,
        # required=True,
        help="Protected attribute we want to improve for this task.",
    )
    parser.add_argument(
        "--clinical_information_path",
        type=str,
        default='clinical_information',
        help="clinical information path",
    )
    parser.add_argument(
        "--foundation_model",
        default='CHIEF',
        help="foundation model used: CHIEF, UNI, GIGAPATH, VIRCHOW2, TITAN",
    )
    parser.add_argument(
        "--max_train_tiles",
        default=None,
        type=int,
        help="Maximum number of tiles to use for training. If None, all samples are used.",
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        default='tile',
        choices=['tile', 'slide'],
        help="Type of feature used for training. Either tile or slide.",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=1,
        help="Number of workers for data loading.")

    parser.add_argument(
        "--task",
        default='mutation',
        help="""
        Type of downstream prediction task:
            - 'mutation': Genetic mutation prediction (default)
            - 'cancer': Cancer type classification
            - 'tumor': Tumor detection
            - Custom: Any column name from clinical data (e.g., 'ER', 'PR', 'HER2')
        
        For backward compatibility, numeric values are also supported:
            - 1: cancer classification
            - 2: tumor detection  
            - 4: mutation prediction
        """,
    )
    parser.add_argument(
        "--genes",
        nargs='+',
        default=None,
        help="For mutation prediction task, specify the genes to classify. If not specified, all genes will be used.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/",
        help="Path to load model weights.",
    )
    parser.add_argument(
        "--inference_output_path",
        type=str,
        default=None,
        help="Path to save the inference results. If not specified, the model is not saved to --model_path.",
    )
        
    parser.add_argument(
        "--weight_path",
        type=str,
        default="",
        help="Path to stage 1 pretrained weights.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Epochs for training."
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Patience for early stopping. If not specified, no early stopping is used."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for sampling images."
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=None,
        help="Batch size for evaluation. If not specified, the same as batch_size.")
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for training."
    )
    parser.add_argument(
        "--curr_fold",
        type=int,
        default=None,
        nargs='*',
        help="For k-fold experiments, current fold(s)."
    )
    parser.add_argument(
        "--partition",
        type=int_or_str,
        default=1,
        help="""
        Data partition method:
         * 1:train/valid/test(6:2:2)
         * 2:k-folds (K=4)
         * fixedKFold: fixed k-folds. Will read the partition from the clinical information file.
        """
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for data partition."
    )
    parser.add_argument(
        "--train_method",
        type=str,
        default="baseline",
        help="Training method for the model. (should be specified in the configs/train_configs.yaml file)"
    )
    parser.add_argument(
        "--reweight",
        action='store_true',
        help="Sample a balanced dataset."
    )
    parser.add_argument(
        "--reweight_cols",
        nargs='+',
        choices=['sensitive', 'label'],
        default=['sensitive', 'label'],
        help="Columns to reweight the dataset. Default is ['sensitive', 'label'].")
    
    parser.add_argument(
        "--reweight_method",
        type=str,
        choices=['none', 'undersample', 'oversample', 'weightedsampler'],
        default='undersample',
        help="Method for reweighting the dataset.")
        
    parser.add_argument(
        "--fair_lambda",
        type=float,
        nargs='+',
        default=[0.5],
        help="weighting parameters for fairness losses. the number of elements should be the same as the number of fairness losses in --constraint."
    )
    parser.add_argument(
        "--l2_lambda",
        type=float,
        default=0.0,
        help="L2 regularization parameter."
    )
    parser.add_argument(
        "--class_loss",
        type=str,
        default='CrossEntropy',
        choices=['CrossEntropy', 'QALY'],
        help="Loss function for classification")
    parser.add_argument(
        "--selection",
        type=str,
        default=None,
        help="Model selection strategy for fine-tuning."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cpu or cuda",
    )
    parser.add_argument(
        "--acc_grad",
        type=int,
        default=1,
        help="Accumulation gradient."
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Dropout rate for the model"
    )
    parser.add_argument(
        "--scheduler_gamma",
        type=float,
        default=1,
        help="Gamma for scheduler"
    )
    parser.add_argument(
        "--scheduler_step",
        type=float,
        default=10,
        help="Steps for scheduler"
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=1.0,
        help="Split ratio for training set"
    )
    parser.add_argument(
        "--data_source",
        type=str,
        default='TCGA',
        choices=[
            'TCGA', 'CPTAC','DFCI', 'RoswellPark','EBrain','BWH_FS',
            'BWH_PM', 'Mayo', 'PLCO', 'UPenn_PM', 'UPenn_FS','TVGH_PM','TVGH_FS'],
        help="embeddings data source used to train the model. If not specified, --feature_paths must be provided"
    )
    parser.add_argument(
        "--embeddings_base_path",
        type=str,
        default='./data/features/',
        help="Base path for WSI feature embeddings (user must provide their own data path)"
    )

    parser.add_argument(
        "--feature_paths",
        type=str,
        nargs='+',
        default=None,
        help="feature paths for different data sources. Can be a list of paths"
    )
    parser.add_argument(
        "--slide_type",
        type=str,
        default='PM',
        choices=['PM', 'FS','mixed'],
        help="type of slide, either PM or FS"
    )
    # method:Literal[None,'none','MicroBAcc','MacroBAcc']=None):
    parser.add_argument(
        "--cutoff_method",
        type=str,
        choices=list(typing.get_args(CUTOFF_METHODS)),
        default='none',
        help="Cut off method for binary classification")

    parser.add_argument(
        "--input_feature_length",
        type=int,
        default=768,
        help="input feature length of different foundation model. 768 for CHIEF, 1024 for UNI, 1536 for GigaPath, 2560 for VIRCHOW2"
    )
    parser.add_argument(
        "--latent_dims",type=int,nargs='*',
        default=[256, 128],
        help="Latent dimensions for the MLP model. Default is [256, 128].")
    parser.add_argument(
        "--use_sensitive",
        type=bool,
        default=False,
        help= "add sensitive attribute as input"
        
    )
    
    parser.add_argument(
        "--inference_only",
        action="store_true",
        help="perform inference only for task 1, 2 and 3 in main.py"
    )
    parser.add_argument(
        "--inference_mode",
        type=str,
        nargs='+',
        default=['test'],
        choices=['valid', 'test','train','all'],
        help="Partition to perform inference on. Default is test."
    )
    parser.add_argument(
        "--inference_output_features",
        action="store_true",
        default=False,
        help="output features during inference"
    )
    parser.add_argument(
        "--sig_test_only",
        action="store_true",
        help="perform significance test only from preexisting results"
    )
    parser.add_argument(
        "--no_sig_test",
        action="store_true",
        help="do not perform significance test"
    )
    parser.add_argument(
        "--sig_agg_method",
        type=str,
        choices=['concatenate','fisher','groupwise'],
        default='concatenate',
        help=""" method to aggregate p-values. Options are:
            - 'concatenate': concatenate the input data and run a single statistical test
            - 'fisher', 'pearson', 'tippett', 'stouffer', 'mudholkar_george': methods to combine p-values, see scipy.stats.combine_pvalues for details
            - 'groupwise': estimate the fairness metrics first, and then perform bootstraping on population level
        """
    )
    parser.add_argument(
        "--sig_n_bootstraps",
        type=int,
        default=1000,
        help="number of bootstraps for significance test"
    )
    
    parser.add_argument(
        "--magnification",
        type=int,
        default=20,
        help="Magnification of the patches in the training dataset"
    )
    # parser.add_argument(
    #     "--inference_validation",
    #     action="store_true",
    #     help="Also perform inference on the validation set"
    # )
    parser.add_argument(
        "--stain_norm",
        action="store_true",
        help="indicate if the patches in the training dataset were stain normalised"
    )
    parser.add_argument(
        "--dataset_config_yaml",
        type=str,
        default="configs/dataset_configs.yaml",
        help="""yaml file for dataset configurations""")
    parser.add_argument(
        "--attribute_map_yaml",
        type=str,
        default="configs/attribute_mappings.yaml",
        help="""yaml file for mapping demographic attributes""")
    # just for debugging
    parser.add_argument(
        "--show_testing_when_training",
        action="store_true",
        default=False,
        help="show testing results when training"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=False,
        help="skip if the results exists (for inference and significance test)"
    )
    parser.add_argument(
        "--skip_fair",
        action="store_true",
        default=False,
        help="When training fairness correction, skip the tasks that are already fair in the baseline"
    )
    parser.add_argument(
        "--skip_fair_metrics",
        type=str,
        nargs='+',
        default=["EOddAbs","EOddMax","EOpp0","EOpp1","EBAcc",'AUCDiff'],
        help="if --skip_fair is True, specify the fairness metrics to check for fairness"
    ),
    parser.add_argument(
        "--skip_poor_performance",
        action="store_true",
        default=False,
        help="skip the tasks that have poor performance in the baseline"
    )
    parser.add_argument(
        "--reinit_func",
        type=str,
        default=None,
        help="function to reinitialize the model. Default is None (no reinitialization)"
    )
    parser.add_argument(
        "--reinit_kwargs",
        type=str,
        default=None,
        help="gain for reinitialization (if applicable)"
    )
    
    # Demographic-informed agent parameters
    parser.add_argument(
        "--use_demographic_agent",
        action="store_true",
        help="Enable demographic-informed patch filtering agent"
    )
    parser.add_argument(
        "--demographic_strategy",
        type=str,
        default='attention_confidence',
        choices=['attention_confidence', 'random', 'none', 'unified'],
        help="Agent strategy: attention_confidence (smart filtering), random (control), none (baseline), unified (enhanced multi-signal ensemble)"
    )
    parser.add_argument(
        "--demographic_base_percentile",
        type=int,
        default=25,
        choices=[10, 15, 25, 50],
        help="Base percentage of patches to filter (10, 15, 25, or 50)"
    )
    parser.add_argument(
        "--demographic_adaptive",
        action="store_true",
        help="Enable adaptive filtering: scale percentile by slide confidence"
    )
    parser.add_argument(
        "--demographic_use_correctness_weighting",
        action="store_true",
        help="Weight confidence by prediction correctness"
    )
    parser.add_argument(
        "--demographic_use_v6_routing",
        action="store_true",
        default=True,
        help="Enable multi-factor routing: considers group imbalance, demographic model accuracy, and training progress (default: True)"
    )
    parser.add_argument(
        "--demographic_attribute",
        type=str,
        default=None,
        choices=['Age', 'Race', 'Sex'],
        help="Demographic attribute for agent (Age, Race, or Sex)"
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    # check if args.task can be converted to int. If not, it is a custom task
    try:
        args.task = int(args.task)
    except:
        pass
    ## override args.feature_type based on the foundation model
    args.feature_type = FOUNDATION_MDL_FEATURE_TYPE_MAP[args.foundation_model]
    # Override the default training configs with the specified train method
    # Set training configuration defaults (formerly in train_configs.yaml)
    if not hasattr(args, 'reweight'):
        args.reweight = False
    if not hasattr(args, 'selection'):
        args.selection = 'AUROC'
    if not hasattr(args, 'pretrained'):
        args.pretrained = False
    if not hasattr(args, 'finetune_layer_names'):
        args.finetune_layer_names = None
    ## parse reinit_kwargs
    if isinstance(args.reinit_kwargs, str):
        args.reinit_kwargs = eval(args.reinit_kwargs)   
    ## args.curr_fold is None if its length is 0
    if args.curr_fold is not None and len(args.curr_fold) == 0:
        args.curr_fold = None
    ##
    if args.eval_batch_size is None:
        print("Eval batch size is not specified. Using the same as batch size.")
        args.eval_batch_size = args.batch_size
    if args.inference_output_path is None:
        print(f"Inference output path is not specified. Using the same as model_path.")
        args.inference_output_path = args.model_path
    ## if reweight is false, set reweight_method to none
    if not args.reweight:
        print("Reweight is false. Setting reweight_method to none.")
        args.reweight_method = 'none'
    ## processing fair_attr
    if args.fair_attr is not None:
        try:
            args.fair_attr  = eval(args.fair_attr)
        except:
            raise ValueError(f"fair_attr should be a dictionary. Is {args.fair_attr} instead")
    ## Set default values for legacy parameters (constraint, fair_lambda, pretrained)
    if not hasattr(args, 'constraint'):
        args.constraint = []
    if not hasattr(args, 'fair_lambda'):
        args.fair_lambda = []
    if not hasattr(args, 'pretrained'):
        args.pretrained = False
    ## handle multiple constraints and fairness lambdas
    if isinstance(args.constraint, str):
        args.constraint = [args.constraint]
    if isinstance(args.fair_lambda, (int, float)):
        args.fair_lambda = [float(args.fair_lambda)]
    if len(args.constraint) > 0 and len(args.fair_lambda) > 0:
        assert len(args.constraint) == len(args.fair_lambda), ValueError("Number of constraints and fairness lambdas should be the same.")
    ##
    if print:
        print("=========\tArguments\t=========")
        for arg in vars(args):
            print(f"\t{arg}:\t{getattr(args, arg)}")
        print("========\tEnd of Arguments\t=======")
    return args

def save_args(args, dir_to_save, filename='args.json'):
    # Convert Namespace to dictionary
    args_dict = vars(args)
    file_path = os.path.join(dir_to_save, filename)
    # Export args_dict to a JSON file
    with open(file_path, 'w') as json_file:
        json.dump(args_dict, json_file, indent=4)

    print(f"Arguments have been saved to {file_path}")

def generate_dataset(args_in,inference_mode=None):
    ## load dataset configuration
    args = copy.deepcopy(args_in)
    with open(args.dataset_config_yaml, 'r') as stream:
        try:
            dataset_configs = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    dataset_config = dataset_configs[args.data_source]
    ##
    strEmbeddingPath = construct_wsi_embedding_path(args)
    if 'geneType' not in args:
        args.geneType = ''
    if 'geneName' not in args:
        args.geneName = ''
    assert args.data_source in dataset_configs.keys(), f"Dataset {args.data_source} not found in the dataset configuration yaml ({args.dataset_config_yaml})."
    ## get dataset generator class from string
    generator_name = dataset_config['generator']
    dataset_generator = globals()[generator_name]
    ## define default dataset arguments
    intDiagnosticSlide_mapping = {"PM": 1, "FS": 0, "mixed": None}
    ## If inference mode is specified to be 'all', set partition to 0 (no partition)
    if inference_mode == 'all':
        args.partition=0 
    
    dataset_args = {
        'cancer': args.cancer,
        'sensitive': args.fair_attr,
        'fold': args.partition,
        'task': args.task,
        'seed': args.seed,
        'feature_type': args.feature_type,
        'strEmbeddingPath': strEmbeddingPath,
        'intDiagnosticSlide': intDiagnosticSlide_mapping[args.slide_type],
        # 'strClinicalInformationPath': args.strClinicalInformationPath,
        # 'age_col': args.age_col,
        'attribute_map_yaml': args.attribute_map_yaml,
        'geneType': args.geneType,
        'geneName': args.geneName,
    }
    ## get dataset-specific arguments
    dataset_new_args = dataset_config['args']
    ## update dataset-specific arguments with the provided arguments
    print(f"Overriding dataset-specific arguments with the provided arguments:")
    for k, v in dataset_new_args.items():
        print(f" \t{k}:\t{v}")
    dataset_args.update(dataset_new_args)
    ## instantiate the dataset generator
    data = dataset_generator(**dataset_args)
    
    return data

def should_skip_task(args, check_dir,
                     mode:Literal['train','inference','sig_test_bias','sig_test_improved']='train',
                     inference_mode:Literal['valid', 'test','train','all']='test'):
    '''
    check if the task should be skipped
    based on the following arguments:
    --skip_fair, --skip_fair_metrics, --skip_poor_performance, --skip_existing
    check_dir: directory to check for existing results
    mode: type of task. 'train' or 'inference' or 'sig_test_bias' or 'sig_test_improved'
    inference_mode: partition to perform inference on. Default is test.
    '''
    SHOULD_SKIP = False
    
    inf_mode_prefix_map = {'valid': 'valid_', 'test': '', 'train': 'train_', 'all': 'all_'}
    inf_mode_prefix = inf_mode_prefix_map[inference_mode]
    max_index=None
    for max_index, _ in test_os_settings(args):
        break
    exist_check_paths_map = {
        'train': join(check_dir, f"inference_results.csv"),
        'inference': join(check_dir, f"{inf_mode_prefix}inference_results.csv"),
        'sig_test_bias':  join(check_dir,f"{max_index}_bootstrapTest_{args.sig_agg_method}_metrics.csv"),
        'sig_test_improved':  join(check_dir,f"{max_index}_bootstrapTest_{args.sig_agg_method}_improvement.csv")
    }
    ## check skip_existing
    if args.skip_existing:
        print("--skip_existing is set. Checking existing results...")
        ## check if the results exist
        check_results_path = exist_check_paths_map[mode]
        if os.path.exists(check_results_path):
            print(f"Results exist: {check_results_path}. Skip training.")
            SHOULD_SKIP=True
    ## if not baseline, also check skip_fair & skip_poor_performance
    if args.pretrained:
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
            SHOULD_SKIP=True
        baseline_dir = dirname(baseline_results_folders[0])
        if args.skip_poor_performance:
            print("--skip_poor_performance is set. Checking baseline performance...")
            check_results_path = join(baseline_dir, f"{baseline_max_index}_bootstrapTest_{args.sig_agg_method}_p_AUC.csv")
            if not os.path.exists(check_results_path):
                print(f"p_AUC not found: {check_results_path}. Skip training.")
                SHOULD_SKIP=True
            df = pd.read_csv(check_results_path,index_col=0).T.iloc[0]
            AUC = df['AUROC']
            p_AUC = df['p_value']
            if p_AUC > 0.05:
                print(f"Insufficient performance in baseline: AUC={AUC:.3f} (p={p_AUC:.3f}). Skip training.")
                SHOULD_SKIP=True
        if args.skip_fair:
            print(f"--skip_fair is set. Checking baseline fairness ({', '.join(args.skip_fair_metrics)})...")
            check_results_path = join(baseline_dir, f"{baseline_max_index}_bootstrapTest_{args.sig_agg_method}_p_biased.csv")
            if not os.path.exists(check_results_path):
                print(f"Fairness results not found: {check_results_path}. Skip training.")
                SHOULD_SKIP=True
            df = pd.read_csv(check_results_path,index_col=0)
            check_pvals = df[args.skip_fair_metrics].iloc[0].to_numpy()
            min_pval = np.min(check_pvals)
            if min_pval > 0.05:
                print(f"No bias in baseline: min p-value={min_pval:.3f}. Skip training.")
                SHOULD_SKIP=True
    return SHOULD_SKIP

        





def get_model(args,num_classes):
    if args.feature_type == 'tile':
        if args.task in [1,2,4] or type(args.task) == str:
            model_class = ClfNetwSensitive if args.use_sensitive else ClfNet
            model = model_class(featureLength=args.input_feature_length,latent_dims=args.latent_dims,
                            classes=num_classes, dropout=args.dropout)
    elif args.feature_type == 'slide':
        model_class = MLPwSensitive if args.use_sensitive else MLP
        model = model_class(featureLength=args.input_feature_length,latent_dims=args.latent_dims,
                        classes=num_classes, dropout=args.dropout)
    return model
def construct_wsi_embedding_path(args):
    if args.foundation_model == 'TITAN':
        if args.data_source == 'TCGA':
            # User must provide path via --embeddings_base_path argument
            cancer_to_path = os.path.join(args.embeddings_base_path, 'TCGA_TITAN_features.pkl')
        else:
            raise notImplementedError(f"Data source {args.data_source} not supported for TITAN model.")
        return cancer_to_path
        
    if args.stain_norm:
        stain_norm_str = "(stain_norm)"
    else:
        stain_norm_str = ""
    # Use user-provided base path
    mag = f"{args.magnification}X"
    cancer_to_path = {}

    for cancer in args.cancer:
        cancer = cancer.upper()
        if cancer == "COADREAD":  # handle task4 naming convention
            cancer_to_path["COAD"] = _get_feature_path(
                args, "COAD", mag, stain_norm_str)
            cancer_to_path["READ"] = _get_feature_path(
                args, "READ", mag, stain_norm_str)
            continue
        cancer_to_path[cancer] = _get_feature_path(
            args, cancer, mag, stain_norm_str)

    return cancer_to_path

def _get_feature_path(args, cancer, mag, stain_norm_str):
    base = args.embeddings_base_path
    if args.data_source == "TCGA":
        SUBTYPE_MAP = {
            'BRCA-IDC': 'BRCA',
            'BRCA-ILC': 'BRCA',
            'BRCA-IDCMIXED': 'BRCA',
            'BRCA-ILCMIXED': 'BRCA',
            'LUAD-BAC': 'LUAD',
            'LUAD-NBAC': 'LUAD',
            'UCEC-SEROUS': 'UCEC',
            'UCEC-NONSEROUS': 'UCEC'
        }
        cancer = SUBTYPE_MAP.get(cancer, cancer)
        
        
        if args.slide_type == 'mixed':
            return [join(base, f"TCGA-{cancer}-PM/{args.foundation_model}/{mag}/pt_files{stain_norm_str}"), \
                join(base, f"TCGA-{cancer}-FS/{args.foundation_model}/{mag}/pt_files{stain_norm_str}")]
        return join(base, f"TCGA-{cancer}-{args.slide_type}/{args.foundation_model}/{mag}/pt_files{stain_norm_str}")
    elif args.data_source == "RoswellPark":
        return join(base, f"RoswellParkCancerCenter/{args.foundation_model}/{mag}/pt_files{stain_norm_str}")
    elif args.data_source == "CPTAC":
        # mapping harmonizing the cancer names between TCGA and CPTAC
        # (key: TCGA name, value: CPTAC name)
        CANCER_NAME_MAP = {
            'LUSC': 'LSCC',
            'KIRC': 'CCRCC',
            'HNSC': 'HNSCC',
            'PAAD': 'PDA'
        } 
        cancer_renamed = CANCER_NAME_MAP.get(cancer, cancer) # rename LUSC to LSCC
        return join(base, f"CPTAC_{cancer_renamed}/{args.foundation_model}/{mag}/pt_files{stain_norm_str}")
    elif args.data_source == "DFCI":  # organ names are used to name the feature files for DFCI dataset
        DFCI_CANCER_MAP = {
            'BRCA':'BREAST',
            'BRCA-IDC':'BREAST',
            'BRCA-ILC':'BREAST',
            'COAD': 'COLORECTAL',
            'READ': 'COLORECTAL',
            'GBM': 'BRAIN',
            'LGG': 'BRAIN',
            'KICH': 'RENAL',
            'KIRC': 'RENAL',
            'KIRP': 'RENAL',
            'LUAD': 'LUNG',
            'LUSC': 'LUNG',
        }
        cancer_renamed=  DFCI_CANCER_MAP.get(cancer, cancer)
        return join(base, f"DFCI_{cancer_renamed}/{args.foundation_model}/{mag}/pt_files{stain_norm_str}")
    elif args.data_source == "EBrain":
        # raise NotImplementedError("EBrain dataset is not supported yet.")
        CANCER_NAME_MAP = {
            'LGG': [
                'Diffuse-astrocytoma_IDH-mutant',
                'Diffuse-astrocytoma_IDH-wildtype',
                'Oligodendroglioma_IDH-mutant-and-1p-19q-codeleted',
                'Pilocytic-astrocytoma'
            ],
            'GBM': [
                'Glioblastoma_IDH-mutant',
                'Glioblastoma_IDH-wildtype'
            ]   
        }
        cancer_renamed = CANCER_NAME_MAP.get(cancer, cancer) # rename LUSC to LSCC
        paths = [join(base, f"{c}/{args.foundation_model}/{mag}/pt_files{stain_norm_str}") for c in cancer_renamed]
        return paths
    elif args.data_source == 'Mayo':
        return join(base, f"MayoBrain/{args.foundation_model}/{mag}/pt_files{stain_norm_str}")
    elif args.data_source == 'PLCO':
        CANCER_NAME_MAP = {
            #
            'COAD': 'Colorectal',
            'READ': 'Colorectal',
            #
            'LUAD': 'Lung',
            'LUSC': 'Lung',
            'LUAD-BAC': 'Lung',
            'LUAD-nBAC': 'Lung',
            #
            'OV-serous': 'Ovarian',
            'OV-nonserous': 'Ovarian',
        }
        cancer_renamed = CANCER_NAME_MAP.get(cancer, cancer) # rename LUSC to LSCC
        paths = [join(base, f"{c}/{args.foundation_model}/{mag}/pt_files{stain_norm_str}") for c in cancer_renamed]
        return join(base, f"PLCO_{cancer_renamed}/{args.foundation_model}/{mag}/pt_files{stain_norm_str}")
    elif args.data_source in ['BWH_FS', 'BWH_PM','UPenn_PM', 'UPenn_FS']:
        return join(base, f"GBMFrozen/{args.foundation_model}/{mag}/pt_files{stain_norm_str}")
    ## BWH_FS, BWH_PM, Mayo, PLCO, UPenn_PM, UPenn_FS
    elif args.data_source == 'TVGH_PM':
        return join(base, f"TVGH_brain_PM/{args.foundation_model}/{mag}/pt_files{stain_norm_str}")
    elif args.data_source == 'TVGH_FS':
        return join(base, f"TVGH_brain_FS/{args.foundation_model}/{mag}/pt_files{stain_norm_str}")


    else:
        raise NotImplementedError(f"Data source {args.data_source} not supported yet.")
        

def train_os_settings(args):
    max_index = None
    reweight_str = f"_finetune_{args.train_method}_{'-'.join(args.constraint)}" if args.pretrained else ""
    cancer_folder = f'{args.task}_{"_".join(args.cancer)}'
    dir = join(args.model_path,
               f"{cancer_folder}_{args.partition}{reweight_str}/")
    os.makedirs(dir, exist_ok=True)

    folder_names = os.listdir(dir)
    subfolders = [folder for folder in folder_names if os.path.isdir(
        os.path.join(dir, folder))]
    if not subfolders:
        max_index = 1
    else:
        model_indexes = [int(name.split('_')[0]) for name in subfolders]
        max_index = max(model_indexes)
        if args.partition == 1:
            max_index += 1
        if args.partition in [2,'fixedKFold'] and args.curr_fold == 0:
            max_index += 1

    return max_index, reweight_str

def test_os_settings(args):
    if args.task != 4:
        max_index = None
        reweight_str = f"_finetune_{args.train_method}_{'-'.join(args.constraint)}" if args.pretrained else ""
        cancer_folder = f'{args.task}_{"_".join(args.cancer)}'
        dir = join(args.model_path,
                f"{cancer_folder}_{args.partition}{reweight_str}/")
        os.makedirs(dir, exist_ok=True)

        folder_names = os.listdir(dir)
        subfolders = [folder for folder in folder_names if os.path.isdir(
            os.path.join(dir, folder))]
        # assert len(subfolders) > 0, f"No model folders found in {dir}"
        # if subfolders is empty, return without yielding
        if len(subfolders) == 0:
            return
        model_indexes = [int(name.split('_')[0]) for name in subfolders]
        max_index = max(model_indexes)
        # search the subfolders with the max index
        subfolders = [folder for folder in subfolders if folder.startswith(str(max_index))]
        maxidx_folders = []
        for fold in range(N_FOLDS):
            # fold_folders = [folder for folder in subfolders if folder.endswith(f"_{fold}")]
            fold_folders = [folder for folder in subfolders if folder.replace(reweight_str,'').endswith(f"_{fold}")]
            if len(fold_folders) == 0:
                print(f"No model folders found for fold {fold} in {dir}")
                fold_failed=True
                break
            # assert len(fold_folders) > 0, f"No model folders found for fold {fold} in {dir}"
            maxidx_folders.append(join(dir, fold_folders[0]))
        yield max_index, maxidx_folders
    else:
        results_path = join(args.model_path, args.cancer[0].lower())
        
        for models in os.listdir(results_path):
            # the folders should be in the format of task_cancer_geneType_geneName_freq_partition
            # e.g. 4_brca_Common Genes_CDH1-Percentage_12.2_2
            if len(models.split("_")) == 6:

                geneType = models.split("_")[2]
                geneName = "_".join(models.split("_")[3:-1])
                geneName_short = geneName.split('-')[0]
                if args.gene not in geneName:
                    continue
                reweight_str = f"_finetune_{args.train_method}_{'-'.join(args.constraint)}" if args.pretrained else ""
                cancer_folder = f"{args.task}_{ '_'.join(args.cancer)}_{geneType}_{geneName}_{args.partition}"
                gene_weight_folder = case_insensitive_glob(join(results_path, f"{cancer_folder}{reweight_str}"))[0]
                model_names = os.listdir(gene_weight_folder)
                subfolders = [folder for folder in model_names if os.path.isdir(
                    os.path.join(gene_weight_folder, folder))]
                model_indexes = [int(name.split('_')[0])
                                    for name in subfolders]
                if len(model_indexes) == 0:
                    continue
                max_index = max(model_indexes)
                # search the subfolders with the max index
                subfolders = [folder for folder in subfolders if folder.startswith(str(max_index))]
                
                maxidx_folders = []
                fold_failed=False
                for fold in range(N_FOLDS):
                    fold_folders = [folder for folder in subfolders if folder.replace(reweight_str,'').endswith(f"_{fold}")]
                    if len(fold_folders) == 0:
                        print(f"No model folders found for fold {fold} in {gene_weight_folder}")
                        fold_failed=True
                        break
                    # assert len(fold_folders) > 0, f"No model folders found for fold {fold} in {gene_weight_folder}"
                    maxidx_folders.append(join(gene_weight_folder, fold_folders[0]))
                    
                if fold_failed:
                    continue
                yield max_index, maxidx_folders

def sig_test_os_settings(args):
    ## setting for significance test
    ## same as test_os_settings, except that inference_output_path was used instead of model_path
    max_index_list = []
    maxidx_folders_list = []
    new_args = copy.deepcopy(args)
    if args.inference_output_path != args.model_path:
        new_args.model_path = args.inference_output_path
    for max_index, maxidx_folders in test_os_settings(new_args):
        max_index_list.append(max_index)
        maxidx_folders_list.append(maxidx_folders)
    for max_index, maxidx_folders in zip(max_index_list, maxidx_folders_list):
        if max_index is None:
            continue
        yield max_index, maxidx_folders
    
    # return 

def loss_fn_settings(args, ds):
    # get the loss function for the task
    if args.reweight_method == 'weightedsampler': 
        # if using weighted sampler, set the weight to None
        weight = None
    elif args.task != 3:
        label_counts = ds.df['label'].value_counts(
        ).sort_index().to_numpy()
        label_weight = 1 / label_counts
        label_weight = label_weight / label_weight.sum()
        # label_counts =  label_counts /
        weight = torch.tensor(label_weight).to(args.device)
    else:
        weight = None
    if args.class_loss == 'QALY':
        loss_fn = QALYLoss(
            tp_score=args.tp_score,
            tn_score=args.tn_score,
            fp_score=args.fp_score,
            fn_score=args.fn_score
        )
    elif args.class_loss == 'newQALY':
        # get prevalence
        prevalence = ds.df['label'].mean()
        loss_fn = newQALYLoss(
            prevalence=prevalence,
        )
    elif args.class_loss == 'CrossEntropy':
        # print('CrossEntropyloss')
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    else:
        raise ValueError(f"Loss function {args.class_loss} not supported.")

    #loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    return loss_fn

def task_collate_fn_settings(args):
    # get the collate function for the task
    if args.feature_type == 'tile':
        return collate_fn
    elif args.feature_type == 'slide':
        return slide_level_collate_fn

def train_mutation_os_settings(args):
    reweight_str = f"_finetune_{args.train_method}_{'-'.join(args.constraint)}" if args.pretrained else ""
    directory_path = f"./tcga_pan_cancer/{args.cancer[0]}_tcga_pan_can_atlas_2018/"
    if os.path.isdir(directory_path):
        for types in os.listdir(directory_path):
            if types == 'Common Genes':
                geneType = 'Common Genes'
            elif types == 'Targeted Drugs for Genes':
                geneType = 'Mutated Genes'

            if os.path.isdir(f'{directory_path}/{types}/'):
                for gName in os.listdir(f"{directory_path}/{types}/"):
                    geneName = "_".join(gName.split('_')[1:])
                    cancer_folder = f'{args.task}_{"_".join(args.cancer)}_{geneType}_{geneName}'
                    dir = join(
                        args.model_path, f"{cancer_folder}_{args.partition}{reweight_str}/")
                    os.makedirs(dir, exist_ok=True)

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

    return max_index, reweight_str

def wandb_setup(args, max_index, reweight_str, gene=None):

    cancer_part = "_".join(args.cancer)
    fold_part = f"_fold_{args.curr_fold}" if args.partition in [2,'fixedKFold'] else ""
    gene_part = f"_{gene}" if gene else ""
    job = f"{max_index}_{cancer_part}{gene_part}{fold_part}{reweight_str}"
    task = f"{cancer_part}_{gene_part}{str(args.partition)}{reweight_str}"
    wandb_group = f"{args.task}_{task}"

    wandb.init(project='FAIR-Tuning',
               config={"_service_wait": 6000, **vars(args)},
               name=job,
               group=wandb_group)

    return job, task

def load_pretrained_weights(args, num_classes, max_index):
    model = get_model(args,num_classes)

    # cancer_folder = str(args.task) + "_" + "_".join(args.cancer)
    if args.task != 4: # for all tasks other than mutation prediction
        cancer_folder = f'{args.task}_{"_".join(args.cancer)}_{args.partition}'
    else:
        cancer_folder = join(
            args.cancer[0],  
            f'{args.task}_{"_".join(args.cancer)}_{args.geneType}_{args.geneName}_{args.partition}'
        )

    if args.weight_path != "":
        # weight_path = glob.glob(join(args.model_path, f"{cancer_folder}_{args.partition}",f"{args.weight_path}_*/model.pt"))[0]
        weight_path_sstr = join(
            args.model_path, cancer_folder, f"{args.weight_path}_*/model.pt")
    else:
        if args.partition == 1:
            # weight_path = glob.glob(join(args.model_path, f"{cancer_folder}_{args.partition}", f"{max_index}_*/model.pt"))[0]
            weight_path_sstr = join(
                args.model_path, cancer_folder, f"{max_index}_*/model.pt")

        elif args.partition in [2,'fixedKFold']:
            # weight_path = glob.glob(join(args.model_path, f"{cancer_folder}_{args.partition}", f"{max_index}_*_{args.curr_fold}", "model.pt"))[0]
            weight_path_sstr = join(
                args.model_path, cancer_folder, f"{max_index}_*_{args.curr_fold}", "model.pt")
    weight_paths = glob.glob(weight_path_sstr)
    assert len(weight_paths) > 0, f"Weight path not found: {weight_path_sstr}"
    weight_path = weight_paths[0]

    model.load_state_dict(torch.load(
        weight_path, map_location=args.device), strict=False)
    print(f"Weights path:{weight_path}")
    print("Loaded pretrained weights.")
    return model

def reinitialize_weights(p,reinit_func=None,reinit_kwargs:Dict=None):
    '''
    reinitialize the weights of the model
    inputs:
    - reinit_func: str, the name of the reinitialization function
    - reinit_kwargs: dict, the keyword arguments for the reinitialization function
    '''
    if p.dim() > 1:
        if reinit_func is None:
            # if reinit_func is not specified, do not reinitialize
            return
            # pass
        else:
            ## get the function from the reinit_func string
            parts = reinit_func.split(".")
            module_name = ".".join(parts[:-1])  # Everything before the last part is the module
            reinit_func = parts[-1]  # The last part is the function name
            if module_name == "":
                # if the module name is empty, the function is in the global namespace
                reinit_func = globals()[reinit_func]
            else:
                # Otherwise, import the module and get the function from there
                module = importlib.import_module(module_name)
                reinit_func = getattr(module, reinit_func)
            reinit_func(p, **reinit_kwargs)
            return
    return
def add_noise(p, shrinkage_factor=0.9, perturbation_scale=0.01):
    p.data.mul_(shrinkage_factor)
    p.data.add_(torch.randn_like(p) * perturbation_scale)


def optimizer_settings(args, model):
    parameters_to_update = []
    parameter_names_to_update = []
    n_params = 0

    if not args.pretrained:  # if not pretrained, enable all layers
        for n, p in model.named_parameters():
            p.requires_grad = True
            parameters_to_update.append(p)
            parameter_names_to_update.append(n)
            n_params = n_params + np.prod(p.size())
    else:  # if pretrained, only enable the specified layers in args.finetune_layer_names
        finetune_layer_list = args.finetune_layer_names
        for n, p in model.named_parameters():
            # if any([n.startswith(layer) for layer in finetune_layer_list]):
            if finetune_layer_list == None or any([n.startswith(layer) for layer in finetune_layer_list]):

                p.requires_grad = True
                if args.reinit_func is not None:
                    reinitialize_weights(p, args.reinit_func, args.reinit_kwargs)
                #########################################################
                # Reinitialization
                # if p.dim() > 1:
                #     # xaiver_method
                #     if(args.train_method == "xai_uni"):
                #         torch.nn.init.xavier_uniform_(p)
                #     if(args.train_method == "xai_uni_sqrt2"):
                #         torch.nn.init.xavier_uniform_(p, gain=math.sqrt(2))
                #     if(args.train_method == "xai_norm"):
                #         torch.nn.init.xavier_normal_(p)
                #     if(args.train_method == "xai_norm_sqrt2"):
                #         torch.nn.init.xavier_normal_(p, gain=math.sqrt(2))
                #     # kaiming_method
                #     if(args.train_method == "kai_uni"):
                #         torch.nn.init.kaiming_uniform_(p)
                #     if(args.train_method == "kai_norm"):
                #         torch.nn.init.kaiming_normal_(p)
                #     # Noise
                #     if(args.train_method == "noise"):
                #         shrinkage_factor = 0.9
                #         perturbation_scale = 0.01
                #         p.data.mul_(shrinkage_factor)
                #         p.data.add_(torch.randn_like(p) * perturbation_scale)
                #########################################################
                
                parameters_to_update.append(p)
                parameter_names_to_update.append(n)
                # get total number of elements in the torch tensor
                n_params = n_params + np.prod(p.size())
            else:
                p.requires_grad = False

    optimizer = torch.optim.Adam(parameters_to_update, lr=args.lr, weight_decay=args.l2_lambda)
    print(f"Params to learn:{n_params}")
    # [print(f'\t{n}') for n in parameter_names_to_update]
    return optimizer


def run(args, train_eval_dl, model, num_classes, colour, loss_fn, optimizer=None, epoch=None):
    fair_loss = 0.
    overall_loss = 0.
    group_loss = 0.
    total_train_eval_loss = 0.
    total_fair_loss = 0.
    avg_train_eval_loss = 0.
    avg_fair_loss = 0.
    avg_group_loss = 0.
    avg_overall_loss = 0.
    logits = []
    probs = []
    predictions = []
    predicted_survival_times = []
    true_survival_times = []
    labels = []
    events = []
    senAttrs = []
    caseIds = []
    slideIds = []
    pbar = tqdm(enumerate(train_eval_dl), colour=colour,
                total=len(train_eval_dl), mininterval=10)
    fair_loss_fn = FairLoss(args, loss_fn)
    loss_fn_wrapped = GroupStratifiedLoss(loss_fn)

    for idx, data in pbar:
        if args.feature_type == 'tile':
            wsi_embeddings, lengths, sensitive, label, group, stage, case_id, slide_id = data
            cancer_pred = model(wsi_embeddings.to(args.device), sensitive.to(args.device), lengths)
        elif args.feature_type == 'slide':
            wsi_embeddings, sensitive, label, group, stage, case_id, slide_id = data
            cancer_pred = model(wsi_embeddings.to(args.device), sensitive.to(args.device))
        
        train_eval_loss, group_of_loss = loss_fn_wrapped(
            cancer_pred, torch.nn.functional.one_hot(label, num_classes).float().to(args.device), group)
        train_eval_loss = train_eval_loss / args.acc_grad
        fair_losses = fair_loss_fn(label, cancer_pred, sensitive)
        # apply the loss lambda's
        train_eval_loss = train_eval_loss * args.main_loss_lambda
        fair_lambda = torch.tensor(args.fair_lambda).to(args.device)
        fair_loss = torch.sum(fair_losses * fair_lambda)
        # add the losses
        total_loss = train_eval_loss + fair_loss

        group_losses = sum(group_of_loss.values())

        if model.training:
            total_loss.backward()
            if (idx + 1) % args.acc_grad == 0:
                optimizer.step()
                optimizer.zero_grad()

        if not torch.isnan(total_loss):
            overall_loss += total_loss.detach().cpu().numpy()
        if not torch.isnan(train_eval_loss):
            total_train_eval_loss += train_eval_loss.detach().cpu().numpy()
        # group_loss += group_losses.detach().cpu().numpy() ???
        if not torch.isnan(group_losses):
            group_loss = group_losses.detach().cpu().numpy()
        total_fair_loss += fair_loss

        avg_overall_loss = overall_loss / (idx+1)
        avg_train_eval_loss = total_train_eval_loss / (idx+1)
        avg_fair_loss = total_fair_loss / (idx+1)
        avg_group_loss = group_loss / (idx+1)

        pbar.set_description(
            (f"Iter:{epoch+1:3}/{args.epochs:3} " if epoch is not None else "") +
            f"Avg_loss:{avg_train_eval_loss:.4f} " +
            f"Fair_loss:{avg_fair_loss:.4f} " +
            f"Group_loss:{avg_group_loss:.4f} ", refresh=False)
        # pbar.update()

        if not model.training:
            predictions.append(torch.argmax(
                cancer_pred.detach().cpu(), dim=1).numpy())
            logits.append(cancer_pred.detach().cpu().numpy())
            probs.append(torch.nn.functional.softmax(
                cancer_pred, dim=1).detach().cpu().numpy())
            labels.append(label.detach().cpu().numpy())
            senAttrs.append(sensitive.detach().cpu().numpy())

            caseIds.append(case_id)
            slideIds.append(slide_id)

    if model.training:
        return avg_train_eval_loss, avg_fair_loss, avg_group_loss, avg_overall_loss
    else:
        # Handle empty dataloaders (e.g., when partition=0 and train/val sets are empty)
        if len(predictions) == 0 and len(events) == 0:
            # Return 10 empty values with correct shapes for compatibility with main_genetic.py
            # For classification tasks: predictions (1D), probs (2D with num_classes columns), logits (2D)
            empty_predictions = np.array([])
            empty_probs = np.zeros((0, num_classes))  # Empty 2D array with correct number of columns
            empty_logits = np.zeros((0, num_classes))
            empty_results = (np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), 
                           empty_predictions, empty_probs, empty_logits, [], [])
            return empty_results, avg_train_eval_loss, avg_fair_loss, avg_group_loss, avg_overall_loss
        
        # For survival tasks (task 3)
        if args.task == 3:
            if len(predicted_survival_times) > 0:
                predicted_survival_times = np.concatenate(predicted_survival_times, axis=0)
                true_survival_times = np.concatenate(true_survival_times, axis=0)
                events = np.concatenate(events, axis=0)
            else:
                predicted_survival_times = np.array([])
                true_survival_times = np.array([])
                events = np.array([])
            senAttrs = np.concatenate(senAttrs, axis=0)
            caseIds = [item for sublist in caseIds for item in sublist]
            slideIds = [item for sublist in slideIds for item in sublist]
            return (np.array([]), senAttrs, events, true_survival_times, predicted_survival_times,
                   np.array([]), np.array([]), np.array([]), caseIds, slideIds), avg_train_eval_loss, avg_fair_loss, avg_group_loss, avg_overall_loss
        # For classification tasks (task 1, 2, 4, etc.)
        else:
            predictions = np.concatenate(predictions, axis=0)
            logits = np.concatenate(logits, axis=0)
            probs = np.concatenate(probs, axis=0)
            labels = np.concatenate(labels, axis=0)
            senAttrs = np.concatenate(senAttrs, axis=0)
            caseIds = [item for sublist in caseIds for item in sublist]
            slideIds = [item for sublist in slideIds for item in sublist]
            # Return 10 values for compatibility: labels, senAttrs, events (empty), true_survival_times (empty), 
            # predicted_survival_times (empty), predictions, probs, logits, caseIds, slideIds
            return (labels, senAttrs, np.array([]), np.array([]), np.array([]),
                   predictions, probs, logits, caseIds, slideIds), avg_train_eval_loss, avg_fair_loss, avg_group_loss, avg_overall_loss

def test_folder_setup(args, curr_fold,inference_mode:Literal['valid', 'test','train','all']='test'):
    inf_mode_prefix_map = {'valid': 'valid_', 'test': '', 'train': 'train_', 'all': 'all_'}
    inf_mode_prefix = inf_mode_prefix_map[inference_mode]

    reweight_str = f"_finetune_{args.train_method}_{'-'.join(args.constraint)}" if args.pretrained else ""
    fold_str = f"_fold_{curr_fold}" if curr_fold is not None and args.partition in [2,'fixedKFold'] else ""

    cancer_folder = f'{args.task}_{"_".join(args.cancer)}'

    dir_path = join(args.model_path,
                    f"{cancer_folder}_{args.partition}{reweight_str}")

    model_names = os.listdir(dir_path)
    subfolders = [folder for folder in model_names if os.path.isdir(
        os.path.join(dir_path, folder))]
    # get the model index:
    # the first field is the run ID (e.g. for 2-base-kmsjtt4p_fold_3, 2 is the run ID)
    model_indexes = [int(name.split('_')[0]) for name in subfolders]
    # The run ID increases by 1 for each new run. The max run ID is the most recent run.
    max_index = max(model_indexes) if args.weight_path == "" else int(
        args.weight_path)

    # weight_path = glob.glob(join(dir_path,f"{max_index}_*{fold_str}{reweight_str}/model.pt"))[0]
    weight_path_sstr = join(
        dir_path, f"{max_index}_*{fold_str}{reweight_str}/model.pt")
    weight_paths = glob.glob(weight_path_sstr)
    assert len(weight_paths) > 0, f"Weight path not found: {weight_path_sstr}"
    weight_path = weight_paths[0]

    parent_weight_path = Path(weight_path).parent if args.partition == 1 else Path(
        weight_path).parent.parent
    if args.inference_output_path is None or args.inference_output_path == args.model_path:
        result_path = parent_weight_path / f"{max_index}_result.csv"
        model_names = os.listdir(dir_path)
        fig_path = parent_weight_path / \
            f"{str(max_index) + '_' if args.partition in [2,'fixedKFold'] else ''}survival_curve.png"
        fig_path2 = parent_weight_path / \
            f"{str(max_index) + '_' if args.partition in [2,'fixedKFold'] else ''}survival_curve_stage.png"
        fig_path3 = parent_weight_path / \
            f"{str(max_index) + '_' if args.partition in [2,'fixedKFold'] else ''}survival_curve_black.png"
        kfold_results_path = Path(weight_path).parent  / f"{inf_mode_prefix}inference_results_fold{curr_fold}.csv" if args.partition in [2,'fixedKFold'] else ""
        parent_result_path = Path(result_path).parent
    else:
        cancer_part = "_".join(args.cancer)
        fold_part = f"_fold_{args.curr_fold}" if args.partition in [2,'fixedKFold'] else ""

        outdir_path = join(args.inference_output_path,
                        f"{cancer_folder}_{args.partition}{reweight_str}")
        result_path = join(outdir_path, f"{max_index}_result.csv")
        parent_result_path = Path(result_path).parent
    
        fig_path = parent_result_path / \
            f"{str(max_index) + '_' if args.partition in [2,'fixedKFold'] else ''}survival_curve.png"
        fig_path2 = parent_result_path / \
            f"{str(max_index) + '_' if args.partition in [2,'fixedKFold'] else ''}survival_curve_stage.png"
        fig_path3 = parent_result_path / \
            f"{str(max_index) + '_' if args.partition in [2,'fixedKFold'] else ''}survival_curve_black.png"
        kfold_results_path = parent_result_path / f"{inf_mode_prefix}inference_results_fold{curr_fold}.csv" if args.partition in [2,'fixedKFold'] else ""
    os.makedirs(parent_result_path, exist_ok=True)
    

    return result_path, fig_path, fig_path2, fig_path3, weight_path, kfold_results_path


def test_load_pretrained_weights(args, weight_path, num_classes):
    # if args.task == 1 or args.task == 2 or args.task == 4 or type(args.task) == str:
    #     model = ClfNet(featureLength=args.input_feature_length,
    #                    classes=num_classes)
    # elif args.task == 3:
    #     model = WeibullModel(featureLength=args.input_feature_length)
    model = get_model(args,num_classes)

    model.load_state_dict(torch.load(
        weight_path, map_location=args.device), strict=False)

    return model


def test_save_results(args, num_classes, labels, senAttrs, caseIds, predictions, probs, result_path,inf_mode_prefix=''):
    if num_classes > 2:
        results = FairnessMetricsMultiClass(predictions, labels, senAttrs)
        pd.DataFrame(results).T.to_csv(result_path)
        print(f"Save results to:{result_path}")

    elif num_classes == 2:
        auroc = roc_auc_score_nan(labels, probs)

        # threshold = Find_Optimal_Cutoff(labels,probs,senAttrs,method=args.cutoff_method)
        # predictions = torch.ge(torch.tensor(predictions), threshold).int()
        results = FairnessMetrics(predictions, probs, labels, senAttrs)

        temp = {"AUROC": auroc}
        results = {**temp, **results}
        pd.DataFrame(results).T.to_csv(result_path)
        print(f"Save results to:{result_path}")
        ### save the inference results
        
        # inference_results_path = join(dirname(result_path), "inference_results.csv")
        inference_results_path = join(dirname(result_path), f"{inf_mode_prefix}inference_results.csv")
        os.makedirs(dirname(inference_results_path), exist_ok=True)
        inference_results = pd.DataFrame({
            "prob": probs,
            "pred": predictions,
            "label": labels,
            "sens_attr": senAttrs,
            "patient_id": caseIds,
        })
        inference_results.to_csv(inference_results_path)
        

    # fmtc = Metrics(predictions = predictions, labels = labels, sensitives = senAttrs, projectName = "proposed", verbose = True)
    # markdown = fmtc.getResults(markdownFormat=True)
    # if auroc != 0: markdown += f"{auroc:.4f}|"
    # print(markdown)
    print(pd.DataFrame(results).T)


def test_save_kfold_results(logits, probs, predictions, labels, senAttrs, caseIds, slideIds, kfold_results_path, survival_res=None):
    if not survival_res:
        inference_results = pd.DataFrame({
            "logits": logits,
            "prob": probs,
            "pred": predictions,
            "label": labels,
            "sens_attr": senAttrs,
            "patient_id": caseIds,
            "slide_id": slideIds,
        })
        inference_results["pred"] = inference_results["pred"].astype(int)
        inference_results["label"] = inference_results["label"].astype(int)
    else:
        inference_results = pd.DataFrame({
            "predicted_survival_times": survival_res[0],
            "true_survival_times": survival_res[1],
            "stages": survival_res[2],
            "events": survival_res[3],
            "sens_attr": senAttrs,
            "patient_id": caseIds,
            "slide_id": slideIds,
        })
        inference_results["predicted_survival_times"] = inference_results["predicted_survival_times"].astype(
            float)
        inference_results["true_survival_times"] = inference_results["true_survival_times"].astype(
            float)
        inference_results["stages"] = inference_results["stages"].astype(int)
        inference_results["events"] = inference_results["events"].astype(int)

    # inference_results["sens_attr"] = inference_results["sens_attr"].astype(int)
    inference_results.to_csv(kfold_results_path)


def test_run(args, test_pbar, model, weight_path=None):
    caseIds = []
    slideIds = []
    logits = []
    probs = []
    predictions = []
    labels = []
    events = []
    senAttrs = []
    predicted_survival_times = []
    true_survival_times = []
    stages = []

    for _, data in test_pbar:
        if args.task != 3:
            wsi_embeddings, lengths, sensitive, label, group, case_id, slide_id = data
            # wsi_embeddings, lengths, sensitive, label, group = data
            test_cancer_pred = model(wsi_embeddings.to(
                args.device), sensitive.to(args.device), lengths)
            predictions.append(torch.argmax(
                test_cancer_pred.detach().cpu(), dim=1).numpy())
            labels.append(label.detach().cpu().numpy())
            senAttrs.append(sensitive.detach().cpu().numpy())


            caseIds.append(case_id)
            slideIds.append(slide_id)
            # logits.append(test_cancer_pred.detach().cpu().tolist()[0][1])
            logits.append(test_cancer_pred.detach().cpu().numpy())
            # probs.append(torch.nn.functional.softmax(test_cancer_pred, dim=1).detach().cpu().tolist()[0][1])
            probs.append(torch.nn.functional.softmax(
                test_cancer_pred, dim=1).detach().cpu().numpy())

        else:
            wsi_embeddings, lengths, sensitive, event, time, group, stage, case_id, slide_id = data
            # wsi_embeddings, lengths, sensitive, event, time, group, stage = data
            test_shape_scale = model(wsi_embeddings.to(
                args.device), lengths)
            test_shape, test_scale = test_shape_scale[:,
                                                      0], test_shape_scale[:, 1]

            predicted_survival_time = test_scale * \
                torch.exp(torch.log(time.to(args.device) + 1e-8) / test_shape)

            predicted_survival_times.append(
                predicted_survival_time.detach().cpu().numpy())
            true_survival_times.append(time.detach().cpu().numpy())
            events.append(event.detach().cpu().numpy())
            stages.append(stage.detach().cpu().numpy())
            senAttrs.append(sensitive.detach().cpu().numpy())
            caseIds.append(case_id)
            slideIds.append(slide_id)
    # convert to numpy arrays
    if args.task == 3:
        predicted_survival_time = np.concatenate(
            predicted_survival_times, axis=0)
        true_survival_time = np.concatenate(true_survival_times, axis=0)
        events = np.concatenate(events, axis=0)
        stages = np.concatenate(stages, axis=0)
    else:
        # predictions = np.concatenate(predictions, axis=0)
        labels = np.concatenate(labels, axis=0)
        logits = np.concatenate(logits, axis=0)
        probs = np.concatenate(probs, axis=0)
        probs = probs[:, 1]
        logits = logits[:, 1]

        inpath = dirname(weight_path)
        predictions = get_predictions(probs, inpath, method=args.cutoff_method)
    senAttrs = np.concatenate(senAttrs, axis=0)
    caseIds = [item for sublist in caseIds for item in sublist]
    slideIds = [item for sublist in slideIds for item in sublist]
    return slideIds, caseIds, logits, probs, predictions, labels, senAttrs, stages


def aggregate_df(df,col='ID_col',out_col = None,agg_cols=['prob','logits']):
    ## aggregate columns in agg_cols by mean
    ## for other columns, take the first value
    ## ID_col is the column that is used to aggregate the data
    ## out_col is the new name of the ID_col (if None, keep the same name)
    
    df_agg_avg = df.groupby(col)[agg_cols].mean().reset_index()
    df_agg_other = df.groupby(col).first().reset_index()
    df_agg_other.drop(agg_cols,axis=1,inplace=True)
    df_agg = pd.merge(df_agg_avg,df_agg_other,on=col)
    if out_col is not None:
        df_agg.rename(columns={col:out_col},inplace=True)
    return df_agg
    
def run_AUROC_onesample_bootstrap_test(y_true,y_pred,n_bootstraps = 1000):
    ## y_true: true labels
    ## y_pred: predicted probabilities
    ## baseline: the baseline value to compare the AUROC to
    ## returns: p-value of the one-sample bootstrap test
    auroc = roc_auc_score_nan(y_true,y_pred)
    auc_bootstraps = []
    for i in range(n_bootstraps):
        indices1 = np.random.choice(range(len(y_true)),len(y_true),replace=True)
        indices2 = np.random.choice(range(len(y_true)),len(y_true),replace=True)
        auc_bootstraps.append(roc_auc_score_nan(y_true[indices1],y_pred[indices2]))
    auc_bootstraps = np.array(auc_bootstraps)
    p_value = (np.sum(auc_bootstraps >= auroc)) / (n_bootstraps)
    df = pd.Series({'AUROC':auroc,'p_value':p_value})#.to_frame().T
    return df
    




def run_significance_test(args,ID_col_name='patient_id',inference_mode:Literal['valid', 'test','train','all']='test'):
    if args.no_sig_test:
        print("Skip significance test.")
        return
    
    inf_mode_prefix_map = {'valid': 'valid_', 'test': '', 'train': 'train_', 'all': 'all_'}
    inf_mode_prefix = inf_mode_prefix_map[inference_mode]
    get_gene_name=lambda x: '_'.join(basename(dirname(x)).split('_')[2:5])
    # try:
    if True:
        ## run_bias_test
        # max_index, results_folders = sig_test_os_settings(args)
        for  max_index, results_folders in sig_test_os_settings(args):
            gene_full_name = get_gene_name(results_folders[0])
            # results_csvs = [join(folder, f'inference_results_fold{fold}.csv') for fold, folder in enumerate(results_folders)]
            if max_index is None:
                print("No model found. Skip significance test.")
                continue
            results_csvs = [join(folder, f"{inf_mode_prefix}inference_results_fold{fold}.csv") for fold, folder in enumerate(results_folders)]
            assert all([os.path.exists(csv) for csv in results_csvs]),f"Results csvs not found: {results_csvs}"
            outpath = dirname(dirname(results_csvs[0]))
            
            ## Skip if the results already exist
            # check_results_path = join(outpath,f"{max_index}_bootstrapTest_{args.sig_agg_method}_metrics.csv")
            # if os.path.exists(check_results_path) and args.skip_existing:
            #     print(f"Results exist: {check_results_path}. Skip.")
            if not should_skip_task(args,  outpath, mode='sig_test_bias',inference_mode=inference_mode):
                # continue
            #     pass
            # else:
            
            ##
                dfs = [pd.read_csv(csv) for csv in results_csvs]
                # for backward compatibility (patient_id is the new column name)
                for df in dfs:
                    if 'ID_col' in df.columns:
                        df.rename(columns={'ID_col':ID_col_name},inplace=True)
                    
                ## find the ID column


                ## If doing inference on all data, aggregate all folds into ensemble predictions
                if inference_mode == 'all': 
                    df = pd.concat(dfs)
                    if args.task != 2:
                        df = aggregate_df(df,ID_col_name)
                    dfs = [df]
                else:
                    if args.task != 2:
                        dfs = [aggregate_df(df,ID_col_name) for df in dfs]
                ## estimate if the performance significantly differs from random guessing
                df_micro = pd.concat(dfs)
                df_auc = run_AUROC_onesample_bootstrap_test(df_micro['label'].to_numpy(),df_micro['prob'].to_numpy(),n_bootstraps = args.sig_n_bootstraps)
                print(f"AUROC: {df_auc['AUROC']:.4f} p-value: {df_auc['p_value']:.4f}")
                ##
                
                df_p_worse, df_p_better, fairResult, df_CI = CV_bootstrap_bias_test(\
                    dfs, n_bootstrap=args.sig_n_bootstraps, aggregate_method=args.sig_agg_method)
                postfixs = ['p_AUC','p_biased','p_biased_against_majority','metrics','CI']
                for df,postfix in zip([df_auc,df_p_worse,df_p_better,fairResult,df_CI],postfixs):
                    outname = join(outpath,f"{max_index}_bootstrapTest_{args.sig_agg_method}_{postfix}.csv")
                    df.to_csv(outname)
            
            if args.train_method != "baseline":
                baseline_args = copy.deepcopy(args)
                baseline_args.train_method = "baseline"
                # Set baseline defaults
                baseline_args.reweight = False
                baseline_args.selection = 'AUROC'
                baseline_args.pretrained = False
                baseline_args.finetune_layer_names = None
                
                # baseline_max_index, baseline_results_folders = sig_test_os_settings(baseline_args)
                baseline_max_index, baseline_results_folders = None, None
                for baseline_max_index, baseline_results_folders in sig_test_os_settings(baseline_args):
                    ## find the baseline model with the exact gene name
                    if baseline_max_index is None:
                        print("No baseline model found. Skip significance test.")
                        continue
                    baseline_gene_full_name = get_gene_name(baseline_results_folders[0])
                    if gene_full_name == baseline_gene_full_name or args.task != 4:
                        break
                    else:
                        baseline_max_index, baseline_results_folders = None, None

                
                baseline_results_csvs = [join(folder, f'{inf_mode_prefix}inference_results_fold{fold}.csv') for fold, folder in enumerate(baseline_results_folders)]
                assert all([os.path.exists(csv) for csv in baseline_results_csvs]),f"Results csvs not found: {baseline_results_csvs}"
                
                
                baseline_outpath = dirname(dirname(baseline_results_csvs[0]))
                baseline_check_results_path = join(baseline_outpath,f"{baseline_max_index}_bootstrapTest_{args.sig_agg_method}_metrics.csv")
                improv_check_results_path = join(outpath,f"{max_index}_bootstrapTest_{args.sig_agg_method}_improvement.csv")
                if should_skip_task(args, outpath, mode='sig_test_improved',inference_mode=inference_mode):
                    continue

                # if args.skip_existing:
                #     if os.path.exists(baseline_check_results_path) and os.path.exists(improv_check_results_path):
                #         sens_attr1 = pd.read_csv(improv_check_results_path)['sensitiveAttr']
                #         sens_attr2 = pd.read_csv(check_results_path)['sensitiveAttr']
                #         if sens_attr1.equals(sens_attr2):
                #             print(f"Improvement Results exist: {check_results_path}. Skip.")
                #             continue
                #         else:
                #             print(f"Sensitive attributes differ. Rerun the test.")

                dfs = [pd.read_csv(csv) for csv in results_csvs]

                
                baseline_dfs = [pd.read_csv(csv) for csv in baseline_results_csvs]
                ID_cols = [ID_col_name if ID_col_name in x.columns else 'ID_col' for x in baseline_dfs]# for backward compatibility (patient_id is the new column name)
                if args.task != 2:
                    baseline_dfs = [aggregate_df(df,ID_col,out_col=ID_col_name) for df,ID_col in zip(baseline_dfs,ID_cols)]
                df_improv, df_p_better, df_p_worse = CV_bootstrap_improvement_test(
                    dfs_baseline=baseline_dfs, dfs_corrected=dfs, n_bootstrap=args.sig_n_bootstraps,aggregate_method=args.sig_agg_method,
                    ID_col=ID_col_name)
                
                postfixs = ['improvement','p_improved','p_worsened']
                for df,postfix in zip([df_improv,df_p_better,df_p_worse],postfixs):
                    outname = join(outpath,f"{max_index}_bootstrapTest_{args.sig_agg_method}_{postfix}.csv")
                    df.to_csv(outname)
    # except Exception as e:
    if False:
        print(f"Error in significance test: {e}")
        traceback.print_exc()


    return

                
