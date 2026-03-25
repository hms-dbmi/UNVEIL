import torch
from configs import TrainingConfigs, InferenceConfigs
import json
import copy
import shutil
import os
from typing import Optional
from dataset import setup_dataset
from loss_functions import setup_loss_function
from metrics import MetricManager
from model import setup_model
from optimizers import setup_optimizer
from trainer import Inference, Trainer
from utils.data_models import PTBundle
from utils.data_splitting import (
    train_val_test_cv_split,
    train_val_test_from_file_split,
    train_val_test_random_split
)
from utils.infra import get_device, safe_create_directory
from torch.utils.data import DataLoader, WeightedRandomSampler
from scipy import stats
import pandas as pd

class ExperimentRunner:

    def __init__(self, configs_path: str, save_dir: Optional[str] = None):
        self.configs_path = configs_path
        self.save_dir = save_dir
        self.metrics_tables = []

        # load configs file and create the TrainConfigs
        with open(self.configs_path) as f:
            self.configs = TrainingConfigs(**json.load(f))
            self.original_configs = copy.deepcopy(self.configs)

        # overwrite save_dir if necesary
        if self.save_dir is not None:
            self.configs.save_dir = self.save_dir
            

    def run(self):

        n_attempts = 0
        exception = None

        while n_attempts <= self.configs.n_retries:
            if n_attempts != 0:
                if os.path.exists(self.configs.save_dir):
                    shutil.rmtree(self.configs.save_dir)

            # create experimental directory
            safe_create_directory(dir_path=self.configs.save_dir)

            # store the original configs configs
            with open(os.path.join(self.configs.save_dir, "configs.json"), "w") as f:
                json.dump(self.configs.model_dump(), f, indent=4)

            # display the configs to the terminal
            print("*"*100 + "\nExperiment Configurations:\n\n", self.configs, "\n" + "*"*100)

            try:
                # run single training loop or set up cross validation
                if self.configs.cross_validation_folds is not None:
                    self._run_cross_validation()
                else:
                    self._run_train()

                exception = None
                break

            except Exception as e:
                n_attempts += 1
                exception = e

        if exception is not None:
            print(f"Experiment failed after {n_attempts} attempts.")
            print(f"Error: {exception}")
            raise exception


    def _run_cross_validation(self):

        # keeping track of MetricMangers from each fold
        metric_managers = []

        # load the targets data
        sample_keys = []
        for targets_file in self.configs.train_targets_file_path_list:
            with open(targets_file, 'r') as f:
                sample_keys.extend(list(json.load(f).keys()))

        # loop through cross validation folds
        for fold_idx, split_tuple in enumerate(
            train_val_test_cv_split(
                sample_keys=sample_keys,
                val_from_train_ratio=self.configs.val_from_train_ratio,
                n_splits=self.configs.cross_validation_folds,
                split_regex=self.configs.sample_identification_regex
            )
        ):
            print("\n\n", "-"*100, f"\nCross Validation Fold: {fold_idx+1}\n", "-"*100, "\n")
            # making a directory for trianing fold
            fold_save_dir = os.path.join(self.original_configs.save_dir, f"cv_fold_{fold_idx+1}")
            os.makedirs(fold_save_dir, exist_ok=True)
            self.configs.save_dir = fold_save_dir

            # unpack the data split tuple
            train_samples, val_samples, test_samples = split_tuple

            # store the split in the fold directory and updating configs
            fold_split_path = os.path.join(fold_save_dir, "data_split.json")
            self.configs.train_val_test_split_path = fold_split_path
            with open(fold_split_path, "w") as f:
                data_split = {"train": train_samples, "val": val_samples, "test": test_samples}
                json.dump(data_split, f, indent=4)

            # calling training
            metric_manager = self._run_train()
            metric_managers.append(metric_manager)

        self._summarize_cross_validation_metrics()
            

    def _run_train(self) -> MetricManager:

        print("Experiment Setup: \n")

        # get device on which to run the experiment
        device = get_device()
        print("Device: ", device)

        # load targets data from all files for training
        train_targets_dict = {}
        for targets_file in self.configs.train_targets_file_path_list:
            with open(targets_file, 'r') as f:
                train_targets_dict.update(json.load(f))

        # loading the targets on which to validate/test
        val_test_sample_set = set()
        for targets_file in self.configs.val_test_targets_file_path_list:
            with open(targets_file, "r") as f:
                val_test_sample_set = val_test_sample_set.union(set(json.load(f).keys()))

        # split data into train, val, and test sets
        if self.configs.train_val_test_split_path is not None:
            # using a pre-determined data split
            train_samples, val_samples, test_samples = train_val_test_from_file_split(
                sample_keys=list(train_targets_dict.keys()), 
                split_file_path=self.configs.train_val_test_split_path
            )

        else:
            # randomly splitting
            train_samples, val_samples, test_samples = train_val_test_random_split(
                sample_keys=list(train_targets_dict.keys()),
                validation_size=self.configs.validation_size,
                test_size=self.configs.test_size,
                split_regex=self.configs.sample_identification_regex
            )

        fold_split_path = os.path.join(self.configs.save_dir, "data_split.json")
        with open(fold_split_path, "w") as f:
            data_split = {"train": train_samples, "val": val_samples, "test": test_samples}
            json.dump(data_split, f, indent=4)

        # filterinng for samples on which to run validation and test
        val_samples = [sample for sample in val_samples if sample in val_test_sample_set]
        test_samples = [sample for sample in test_samples if sample in val_test_sample_set]
            
        # initialize the train, val, and test datasets
        train_dataset = setup_dataset(
            dataset_type=self.configs.tr_dataset_type,
            dataset_init_args=self.configs.tr_dataset_init_args,
            feature_dir_path_list=self.configs.features_dir_path_list,
            targets_dict=train_targets_dict,
            sample_keys_set=train_samples,
        )
        print("\nTraining Dataset Info:")
        train_dataset._show_data_stats()

        # arguments for validation and test datasets
        val_test_dataset_type = self.configs.tr_dataset_type if self.configs.val_test_dataset_type \
            is None else self.configs.val_test_dataset_type
        val_test_dataset_init_args = self.configs.tr_dataset_init_args if \
            self.configs.val_test_dataset_init_args is None else self.configs.val_test_dataset_init_args

        # arguments for external validation data set
        ex_val_dataset_type = self.configs.tr_dataset_type if self.configs.ex_val_dataset_type \
            is None else self.configs.ex_val_dataset_type
        ex_val_dataset_init_args = self.configs.tr_dataset_init_args if \
            self.configs.ex_val_dataset_init_args is None else self.configs.ex_val_dataset_init_args

        val_dataset = setup_dataset(
            dataset_type=val_test_dataset_type,
            dataset_init_args=val_test_dataset_init_args,
            feature_dir_path_list=self.configs.features_dir_path_list,
            targets_dict=train_targets_dict,
            sample_keys_set=val_samples
        )
        print("\nValidation Dataset Info:")
        val_dataset._show_data_stats()

        test_dataset = setup_dataset(
            dataset_type=val_test_dataset_type,
            dataset_init_args=val_test_dataset_init_args,
            feature_dir_path_list=self.configs.features_dir_path_list,
            targets_dict=train_targets_dict,
            sample_keys_set=test_samples
        )
        print("\nTest Dataset Info:")
        test_dataset._show_data_stats()

        ex_val_dataset = None
        if self.configs.ex_val_features_dir_path_list:
        
            ex_val_targets_dict = {}
            for path in self.configs.ex_val_targets_file_path_list:
                with open(path, "r") as f:
                    ex_val_targets_dict.update(json.load(f))

            ex_val_dataset = setup_dataset(
                dataset_type=ex_val_dataset_type,
                dataset_init_args=ex_val_dataset_init_args,
                feature_dir_path_list=self.configs.ex_val_features_dir_path_list,
                targets_dict=ex_val_targets_dict,
            )

            print("\nExternal Validation Dataset Info:")
            ex_val_dataset._show_data_stats()

        # initialize dataloaders
        if self.configs.class_balance:
            train_sample_weights = train_dataset.get_sample_weights(
                class_balance=self.configs.class_balance,
                names_balance=self.configs.sample_names_balance,
            )
            train_sampler = WeightedRandomSampler(train_sample_weights, len(train_sample_weights))
            print("\nTrain Sampler:")
            print(train_sampler)
        else: 
            train_sampler = None

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.configs.batch_size,
            shuffle=True if train_sampler is None else False,
            sampler=train_sampler,
            drop_last=True
        )
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=1,
            shuffle=False,
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False,
        )

        if ex_val_dataset:
            ex_val_loader = DataLoader(
            dataset=ex_val_dataset,
            batch_size=1,
            shuffle=False,
        )
        else:
            ex_val_loader = None

        # make metric manager
        metric_manager = MetricManager(
            metrics_list=self.configs.metrics, is_multi_class=self.configs.is_multi_class
        )

        # make model
        model = setup_model(
            model_type=self.configs.model_type, 
            model_init_args=self.configs.model_init_args,
        )
        print("\nModel:\n", model)

        # make loss function
        loss_function = setup_loss_function(
            loss_function=self.configs.loss_function, 
            loss_function_init_args=self.configs.loss_function_init_args,
        )
        print("\nLoss Function: ", loss_function)
        
        # make optimizer
        optimizer = setup_optimizer(
            optimizer_type=self.configs.optimizer_type, 
            optimizer_init_args=self.configs.optimizer_init_args,
            params=model.parameters()
        )
        print("\nOptimizer: \n", optimizer)

        # make a directory for trainer to save models
        model_checkpoint_dir = os.path.join(self.configs.save_dir, "models")
        os.makedirs(model_checkpoint_dir, exist_ok=True)

        # make trainer and run the training
        trainer = Trainer(
            model_checkpoint_dir=model_checkpoint_dir,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            ex_val_loader=ex_val_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            model=model,
            metric_manager=metric_manager,
            n_epochs=self.configs.n_epochs,
            best_model_selector_metrics_list=self.configs.best_model_selector_metrics_list,
            gradient_accumulation_steps=self.configs.gradient_accumulation_steps,
            early_stopping_patience=self.configs.early_stopping_patience,
        )
        
        train_log = trainer.train()

        # saving training log
        train_log_path = os.path.join(self.configs.save_dir, "train_log.json")
        with open(train_log_path, "w") as f:
            json.dump(train_log, f, indent=4)

        # saving recorded metrics
        metrics_save_path = os.path.join(self.configs.save_dir, "metrics.csv")
        metric_manager.save_metrics_table(save_path=metrics_save_path)

        # returning metrics manager (for cross validation analysis if necessary)
        self.metrics_tables.append(metric_manager.metrics_table)

    def _summarize_cross_validation_metrics(self):

        # Combine all metric tables into a single table
        combined_metrics_table = pd.concat(self.metrics_tables)

        # Cast dataset column to str
        combined_metrics_table['dataset'] = combined_metrics_table['dataset'].astype(str)

        # Group by all columns except the 'value' column to get unique rows
        combined_metrics_table_grouped = combined_metrics_table.groupby(combined_metrics_table.columns.difference(['value']).tolist())

        # Function to calculate the mean and 95% CI
        def mean_ci(series):
            mean = series.mean()
            # Calculate the 95% confidence interval
            ci = stats.t.interval(0.95, len(series)-1, loc=mean, scale=stats.sem(series))
            ci_width = (ci[1] - ci[0]) / 2
            return mean, ci_width

        # Apply the function to each group and keep the results in the same row
        result = combined_metrics_table_grouped['value'].agg(['mean', lambda x: mean_ci(x)[1]]).reset_index()

        # Rename the lambda column to '95% CI'
        result.columns = list(result.columns)[:-1] +  ['95% CI']

        # save the summary table in the cross validation directory 
        result.to_csv(self.original_configs.save_dir + "/summary_metrics.csv", index=False)


class InferenceRunner:

    def __init__(self, configs_path: str, save_dir: Optional[str] = None):
        self.configs_path = configs_path
        self.save_dir = save_dir
        self.metrics_tables = []

        # load configs file and create the TrainConfigs
        with open(self.configs_path) as f:
            self.configs = InferenceConfigs(**json.load(f))
            self.original_configs = copy.deepcopy(self.configs)

        # overwrite save_dir if necesary
        if self.save_dir is not None:
            self.configs.save_dir = save_dir

    def run(self):

        # create experimental directory
        safe_create_directory(dir_path=self.configs.save_dir)

        # store the original configs configs
        with open(os.path.join(self.configs.save_dir, "configs.json"), "w") as f:
            json.dump(self.configs.model_dump(), f, indent=4)

        # display the configs to the terminal
        print("*"*100 + "\nInference Configurations:\n\n", self.configs, "\n" + "*"*100)
        
        # run inference
        self.infer()

    def infer(self) -> None:

        print("Inference Setup: \n")

        # get device on which to run the experiment
        device = get_device()
        print("Device: ", device)

        # load targets data from all files for training
        targets_dict = {}
        for targets_file in self.configs.targets_file_path_list:
            with open(targets_file, 'r') as f:
                targets_dict.update(json.load(f))

        # load data split if present
        if self.configs.data_split_path is not None and self.configs.data_split_name is not None:
            with open(self.configs.data_split_path, "r") as f:
                data_split = json.load(f)
            sample_keys_set = data_split[self.configs.data_split_name]
        else: 
            sample_keys_set = None

        # making the dataset
        eval_dataset = setup_dataset(
            dataset_type=self.configs.dataset_type,
            dataset_init_args=self.configs.dataset_init_args,
            feature_dir_path_list=self.configs.features_dir_path_list,
            targets_dict=targets_dict,
            sample_keys_set=sample_keys_set,
        )
        print("\nEvaluation Dataset Info:")
        eval_dataset._show_data_stats()

        # making the data loader
        eval_loader = DataLoader(
            dataset=eval_dataset,
            batch_size=1,
            shuffle=False
        )

        # make metric manager
        metric_manager = MetricManager(
            metrics_list=self.configs.metrics, is_multi_class=self.configs.is_multi_class
        )

        # Load bundle and model or get a model generator for ensembling
        if isinstance(self.configs.pt_bundle_path, str):
            ptbundle = PTBundle.load_bundle(load_path=self.configs.pt_bundle_path, map_location=device)
            model = ptbundle.model
            model_generator = None

        elif isinstance(self.configs.pt_bundle_path, list):
            model_generator = get_model_generator(paths=self.configs.pt_bundle_path, device=device)
            model = None

        # Register forward hooks on the model
        intermediate_results = {}

        import numpy as np

        def get_intermediate_output(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    intermediate_results[name] = output.detach().to("cpu").numpy().tolist()
                elif isinstance(output, (list, tuple)):
                    output = np.array([o.detach().to("cpu").numpy() if isinstance(o, torch.Tensor) else o for o in output])
                    output = np.swapaxes(output, 0, 1)
                    intermediate_results[name] = output.tolist()
                else:
                    intermediate_results[name] = output  # Store as-is if it's not a tensor or list/tuple of tensors
            return hook
        
        if model:
            for name, layer in model.named_children():
                if name in self.configs.intermediate_layers_hooks:
                    layer.register_forward_hook(get_intermediate_output(name))

        # make inference module and run inference
        inference = Inference(
            model=model,
            device=device,
            eval_loader=eval_loader,
            metric_manager=metric_manager,
            intermediate_results_dict=intermediate_results,
            model_generator=model_generator
        )
        inference_log = inference.infer()

        # saving metrics
        metrics_save_path = os.path.join(self.configs.save_dir, "metrics.csv")
        metric_manager.save_metrics_table(save_path=metrics_save_path)

        # saving training log
        infer_log_path = os.path.join(self.configs.save_dir, "inference_log.json")
        with open(infer_log_path, "w") as f:
            json.dump(inference_log, f, indent=4)


def get_model_generator(paths, device):
    for path in paths:
        ptbundle = PTBundle.load_bundle(
            load_path=path, map_location=device
        )
        yield ptbundle.model
