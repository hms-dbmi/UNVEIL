import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from metrics import MetricManager, rank_prediction_quality
from tqdm import tqdm
from pathlib import PosixPath
from utils.data_models import DATASET_TYPES, PTBundle
from typing import Dict, Generator
from collections import defaultdict
from utils.data_utils import dict_avg


class Inference:

    def __init__(
        self,
        model: nn.Module | None,
        device: str | torch.device,
        eval_loader: DataLoader,
        metric_manager: MetricManager,
        intermediate_results_dict: Dict,
        model_generator: Generator | None = None,
    ) -> None:

        assert [model, model_generator].count(None) == 1, \
            "One of [model, model_generator] must not be None."
        
        self.model = model
        self.model_generator = model_generator
        self.device = device
        self.data_loaders = {DATASET_TYPES.EVAL: eval_loader}
        self.metric_manager = metric_manager
        self.intermediate_results_dict = intermediate_results_dict
        self.current_model_label = 'inference_model'

    def infer(self) -> Dict:

        if self.model_generator is None:
            # running evaluation the evaluation dataset
            return self.evaluate(dataset=DATASET_TYPES.EVAL)

        else:
            # running ensemble eval
            inference_logs = []

            for model_idx, model in enumerate(self.model_generator):

                self.current_model_label = f'inference_model_{model_idx+1}'
                self.model = model
                inference_logs.append(self.evaluate(dataset=DATASET_TYPES.EVAL))

            # recollect the predictions and targets from the prediction log and calc metrics
            sample_ids = list(inference_logs[0]['inferences'].keys())
            ensemble_inference_log = dict_avg([log['inferences'] for log in inference_logs])
            targets = [ensemble_inference_log[k]['target'] for k in sample_ids]
            preds = [ensemble_inference_log[k]['prediction'] for k in sample_ids]

            # calculate metrics
            self.current_model_label = f'inference_model_ensemble'
            self.metric_manager.calculate_metrics(
                y_true=targets,
                y_pred=preds,
                dataset=DATASET_TYPES.EVAL,
                model_epoch_label=self.current_model_label,
                display=True,
            )

            # calculate prediction rankings across the entire ensemble
            if not self.metric_manager.is_multi_class:
                pred_rank = rank_prediction_quality(
                    sample_keys=sample_ids, targets=targets, predictions=preds
                )
            else: 
                pred_rank = "Multi-class prediction rank currently unsupported."

            inference_log = {
                "inferences": ensemble_inference_log,
                "prediction_quality_rank": pred_rank
            }

            return inference_log


    def evaluate(self, dataset: DATASET_TYPES):

        self.pbar = tqdm(total=len(self.data_loaders[dataset]), unit="batch", leave=False)
        self.model.eval()
        
        # keeping track of targets and predictions across all samples
        inference_dict = defaultdict(dict)
        sample_ids = []
        y_list = []
        y_hat_list = []
        intermediate_results = defaultdict(list)

        with torch.no_grad():
            for x_batch, y_batch, sample_id_batch in self.data_loaders[dataset]:

                # calling forward on the model
                y_batch, y_hat_batch = self.model_forward(x_batch=x_batch, y_batch=y_batch)
                
                # Convert y_batch and y_hat_batch from PyTorch tensors to lists
                y_batch = y_batch.to('cpu').detach().numpy().tolist()
                y_hat_batch = y_hat_batch.to('cpu').detach().numpy().tolist()

                # append to global dataset lists
                y_list += y_batch
                y_hat_list += y_hat_batch
                sample_ids += sample_id_batch

                for sample_id, y, y_hat in zip(sample_id_batch, y_batch, y_hat_batch):
                    inference_dict[sample_id]['target'] = y
                    inference_dict[sample_id]['prediction'] = y_hat

                for hook, hook_output in self.intermediate_results_dict.items():
                    for sample_id, hook_output in zip(sample_id_batch, hook_output):
                        inference_dict[sample_id][hook] = hook_output

                # Update tqdm progress bar with current loss
                self.pbar.set_postfix({"Dataset": dataset, "Mode": "Evaluation"})
                self.pbar.update()

        # removing the pbar from terminal
        self.pbar.clear()
        self.pbar = None

        # calculating metrics, which are internally saved within the metric manager
        self.metric_manager.calculate_metrics(
            y_true=y_list,
            y_pred=y_hat_list,
            dataset=dataset,
            display=True,
            model_epoch_label=self.current_model_label
        )

        # rank predictions based on quality
        if not self.metric_manager.is_multi_class:
            prediction_ranks = rank_prediction_quality(sample_ids, y_list, y_hat_list)
        else:
            prediction_ranks = "Multi-class prediction rank currently unsupported."

        log_dict = {
            "inferences": inference_dict,
            "prediction_quality_rank": prediction_ranks
        }

        return log_dict
    
    def model_forward(self, x_batch, y_batch): 

        # Move data to the appropriate device
        x_batch = x_batch.type(torch.float32).to(self.device)
        y_batch = torch.Tensor(y_batch).type(torch.float32).to(self.device)

        # Forward pass
        y_hat_batch = self.model(x_batch)

        # adjusting the targets shape
        if self.metric_manager.is_multi_class:
            y_batch = y_batch.squeeze(-1).long()
        else: 
            y_batch = y_batch.view(y_hat_batch.shape)

        return y_batch, y_hat_batch
    


class Trainer(Inference):

    def __init__(
        self,
        model_checkpoint_dir: str,
        device: str | torch.device,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader | None,
        loss_function: nn.Module,
        optimizer: Optimizer,
        metric_manager: MetricManager,
        gradient_accumulation_steps: int,
        n_epochs: int,
        model: nn.Module,
        best_model_selector_metrics_list: str,
        early_stopping_patience: int | None,
        ex_val_loader: DataLoader | None = None,
        intermediate_results_dict: Dict = {},
    ) -> None:
        self.model_checkpoint_dir = PosixPath(model_checkpoint_dir)
        self.device = device
        self.data_loaders = {
            DATASET_TYPES.TRAIN: train_loader,
            DATASET_TYPES.VAL: val_loader,
            DATASET_TYPES.TEST: test_loader,
            DATASET_TYPES.EXTERNAL_VAL: ex_val_loader,
        }
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.metric_manager = metric_manager
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.n_epochs = n_epochs
        self.model = model.to(self.device)
        self.best_model_selector_metrics_list = best_model_selector_metrics_list
        self.early_stopping_patience = early_stopping_patience
        self.intermediate_results_dict = intermediate_results_dict

        # fields used during training
        self.pbar = None
        self.best_model_checkpoint_files = set()
        self.current_model_label = None

        # train log
        self.train_log = {}

        # current train step count
        self.train_step_count = 0

        # keeping track of last time best model was updated
        self.last_model_update_epoch = None
        

    def train(self) -> Dict:

        for epoch in range(1, self.n_epochs+1):
            print("\n" + "*"*100)

            self.current_model_label = epoch

            # training 
            avg_train_loss = self.train_step()
            print(f"\nEpoch {epoch} training loss: {avg_train_loss}")

            # evaulate model on the training set
            self.evaluate(dataset=DATASET_TYPES.TRAIN)

            # evaulate model on the validation set
            self.evaluate(dataset=DATASET_TYPES.VAL)

            # evaulate model on the test set 
            # NOTE: this is for retrospective model picking strategy!
            self.evaluate(dataset=DATASET_TYPES.TEST)

            # iterate over each selector metric to see if a new model needs to be saved
            for selector_metric in self.best_model_selector_metrics_list:

                # get the best epochs based on current metric (on val set)
                best_epochs_dict = self.metric_manager.get_best_epochs(
                    dataset=DATASET_TYPES.VAL, metric=selector_metric
                )

                # iterate over each target idx and its best val epoch 
                for target, best_epoch in best_epochs_dict.items():
                    if int(best_epoch) == epoch:
                        self.last_model_update_epoch = epoch
                        model_name = f"target_{target}_best_{selector_metric}"
                        model_file_path = self.model_checkpoint_dir / f"{model_name}.pt"
                        self.best_model_checkpoint_files.add(model_file_path)
                        ptbundle = PTBundle(model=self.model, optimizer=self.optimizer)
                        ptbundle.save_bundle(save_path=model_file_path)
                        print(f"Model saved to {str(model_file_path)}")

            # checking if training can be stopped early
            if (self.early_stopping_patience is not None and
                epoch != self.n_epochs and
                epoch - self.last_model_update_epoch >= self.early_stopping_patience
            ):
                print(f"No models have been saved in {self.early_stopping_patience} epochs.")
                print(f"TRAINING IS STOPPING EARLY.")
                break
                
        # show validation metrics for the best model
        for selector_metric in self.best_model_selector_metrics_list:
            self.metric_manager.summarize_best_epoch_metrics(
                dataset=DATASET_TYPES.VAL, metric=selector_metric, display=True
            )

        # evaluate models on the internal test set
        if self.data_loaders[DATASET_TYPES.TEST]:
            self.train_log["test"] = {}
            for model_file_path in self.best_model_checkpoint_files:
                print("\n" + "*" * 100)
                # load model and run evaluate
                self.current_model_label = model_file_path.with_suffix('').name
                print(f"Loading model from {model_file_path}")
                ptbundle = PTBundle.load_bundle(load_path=model_file_path, map_location=self.device)
                self.model = ptbundle.model
                test_log = self.evaluate(dataset=DATASET_TYPES.TEST)
                self.train_log["test"][self.current_model_label] = test_log

        # evaluate models on external validation set
        if self.data_loaders[DATASET_TYPES.EXTERNAL_VAL]:
            self.train_log["external_validation"] = {}
            for model_file_path in self.best_model_checkpoint_files:
                print("\n" + "*" * 100)
                # load model and run evaluate
                self.current_model_label = model_file_path.with_suffix('').name
                print(f"Loading model from {model_file_path}")
                ptbundle = PTBundle.load_bundle(load_path=model_file_path, map_location=self.device)
                self.model = ptbundle.model
                ex_val_log = self.evaluate(dataset=DATASET_TYPES.EXTERNAL_VAL)
                self.train_log["external_validation"][self.current_model_label] = ex_val_log

        return self.train_log


    def train_step(self) -> float:

        self.pbar = tqdm(total=len(self.data_loaders[DATASET_TYPES.TRAIN]), unit="batch", leave=False)

        self.model.train()
        self.optimizer.zero_grad()
        cummulative_loss = 0

        for x_batch, y_batch, _ in self.data_loaders[DATASET_TYPES.TRAIN]:

            # incrementing step counter (for gradient accumulation)
            self.train_step_count += 1

            # calling forward on the model
            y_batch, y_hat_batch = self.model_forward(x_batch=x_batch, y_batch=y_batch)

            # Calculate loss
            loss = self.loss_function(y_hat_batch, y_batch)
            loss.backward()

            # Backward pass and optimization step
            if self.train_step_count % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.train_step_count = 0
            
            # Accumulate loss
            try:
                cummulative_loss += loss.item()
            except:
                print(f"Invalid Loss (OOB?)")
                print("Predictions: ", y_hat_batch)
                print("Targets: ", y_batch)

            # Update tqdm progress bar with current loss
            self.pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Mode": "Training"})
            self.pbar.update()

        # removing the pbar from terminal
        self.pbar.clear()
        self.pbar = None
        
        return cummulative_loss / len(self.data_loaders[DATASET_TYPES.TRAIN])