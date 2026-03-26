from abc import ABC, abstractmethod
from typing import List, Dict, Union
from sklearn.metrics import (
    roc_auc_score,
    mean_squared_error,
    log_loss,
    balanced_accuracy_score,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
)
from tabulate import tabulate
from itertools import combinations
import numpy as np
import pandas as pd

class Metric(ABC):
    def __init__(self, name: str, maximize: bool, is_multi_target: bool):
        """
        Base class for all metrics.
        
        Args:
            name (str): Name of the metric.
            maximize (bool): Whether to maximize or minimize the metric.
            is_multi_target (bool): Whether to run on each output dimention separately (0) or together (1)
        """
        self.name = name
        self.is_multi_target = is_multi_target
        self.maximize = maximize

    @abstractmethod
    def calculate(self, y_true, y_pred):
        """Calculate the metric."""
        pass

    @property
    def should_maximize(self) -> bool:
        """Returns whether this metric should be maximized."""
        return self.maximize
    
class ThreasholdBasedMetric(Metric):

    def __init__(self, name: str, maximize: bool, threshold: float = 0.5, is_multi_target=False):
        """
        Base class for all metrics.
        
        Args:
            name (str): Name of the metric.
            maximize (bool): Whether to maximize or minimize the metric.
            threshold (flaot): what the threashold at which a prediction is considered positive
            is_multi_target (bool): Whether to run on each output dimention separately (0) or together (1)
        """
        self.name = name
        self.maximize = maximize
        self.is_multi_target = is_multi_target
        self.threshold = threshold

    def binarize_predictions(self, predictions):
        prediction_binarized = np.copy(predictions)
        positive_mask = prediction_binarized > self.threshold
        prediction_binarized[positive_mask] = 1.0
        prediction_binarized[~positive_mask] = 0.0
        return prediction_binarized
    
class SampleSize(Metric):
    def __init__(self):
        super().__init__(name="SampleSize", maximize=None, is_multi_target=False)

    def calculate(self, y_true, y_pred):
        return y_true.shape[0]

class MSE(Metric):
    def __init__(self):
        super().__init__(name="MSE", maximize=False, is_multi_target=False)

    def calculate(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred)
    
class AUC(Metric):
    def __init__(self):
        super().__init__(name="AUC", maximize=True, is_multi_target=False)

    def calculate(self, y_true, y_pred):
        return roc_auc_score(y_true, y_pred)
    
class BCE(Metric):
    def __init__(self):
        super().__init__(name="BCE", maximize=False, is_multi_target=False)

    def calculate(self, y_true, y_pred):
        return log_loss(y_true, y_pred)


class Accuracy(ThreasholdBasedMetric):
    def __init__(self, threshold:float = 0.5):
        super().__init__(name="Accuracy", maximize=True, threshold=threshold, is_multi_target=False)

    def calculate(self, y_true, y_pred):
        y_pred_binary = self.binarize_predictions(predictions=y_pred)
        return accuracy_score(y_true, y_pred_binary)

class BalancedAccuracy(ThreasholdBasedMetric):
    def __init__(self, threshold:float = 0.5):
        super().__init__(name="BalancedAccuracy", maximize=True, threshold=threshold, is_multi_target=False)

    def calculate(self, y_true, y_pred):
        y_pred_binary = self.binarize_predictions(predictions=y_pred)
        return balanced_accuracy_score(y_true, y_pred_binary)

class Recall(ThreasholdBasedMetric):
    def __init__(self, threshold:float = 0.5):
        super().__init__(name="Recall", maximize=True, threshold=threshold, is_multi_target=False)

    def calculate(self, y_true, y_pred):
        y_pred_binary = self.binarize_predictions(predictions=y_pred)
        return recall_score(y_true, y_pred_binary)
    
class Precision(ThreasholdBasedMetric):
    def __init__(self, threshold:float = 0.5):
        super().__init__(name="Precision", maximize=True, threshold=threshold, is_multi_target=False)

    def calculate(self, y_true, y_pred):
        y_pred_binary = self.binarize_predictions(predictions=y_pred)
        return precision_score(y_true, y_pred_binary)

class F1(ThreasholdBasedMetric):
    def __init__(self, threshold:float = 0.5):
        super().__init__(name="F1", maximize=True, threshold=threshold, is_multi_target=False)

    def calculate(self, y_true, y_pred):
        y_pred_binary = self.binarize_predictions(predictions=y_pred)
        return f1_score(y_true, y_pred_binary)

class CE(Metric):
    def __init__(self):
        super().__init__(name="CE", maximize=False, is_multi_target=True)

    def calculate(self, y_true, y_pred):
        return log_loss(y_true, y_pred)
    
class Top1Accuracy(Metric):
    def __init__(self):
        super().__init__(name="Top1Accuracy", maximize=True, is_multi_target=True)

    def calculate(self, y_true, y_pred):
        if y_true.shape[-1] >= 2:
            y_true = np.argmax(y_true, axis=1)
        y_pred_classes = np.argmax(y_pred, axis=1)
        return accuracy_score(y_true, y_pred_classes)

class Top1BalancedAccuracy(Metric):
    def __init__(self):
        super().__init__(name="Top1BalancedAccuracy", maximize=True, is_multi_target=True)

    def calculate(self, y_true, y_pred):
        y_pred_classes = np.argmax(y_pred, axis=1)
        return balanced_accuracy_score(y_true, y_pred_classes)


class TP(ThreasholdBasedMetric):
    def __init__(self, threshold:float = 0.5):
        super().__init__(name="TP", maximize=True, threshold=threshold, is_multi_target=False)

    def calculate(self, y_true, y_pred):
        y_pred_binary = self.binarize_predictions(predictions=y_pred)
        # True positives: both y_true and y_pred_binary are 1
        return np.sum((y_true == 1) & (y_pred_binary == 1))


class FP(ThreasholdBasedMetric):
    def __init__(self, threshold:float = 0.5):
        super().__init__(name="FP", maximize=True, threshold=threshold, is_multi_target=False)

    def calculate(self, y_true, y_pred):
        y_pred_binary = self.binarize_predictions(predictions=y_pred)
        # False positives: y_true is 0, but y_pred_binary is 1
        return np.sum((y_true == 0) & (y_pred_binary == 1))


class TN(ThreasholdBasedMetric):
    def __init__(self, threshold:float = 0.5):
        super().__init__(name="TN", maximize=True, threshold=threshold, is_multi_target=False)

    def calculate(self, y_true, y_pred):
        y_pred_binary = self.binarize_predictions(predictions=y_pred)
        # True negatives: both y_true and y_pred_binary are 0
        return np.sum((y_true == 0) & (y_pred_binary == 0))

class FN(ThreasholdBasedMetric):
    def __init__(self, threshold:float = 0.5):
        super().__init__(name="FN", maximize=True, threshold=threshold, is_multi_target=False)

    def calculate(self, y_true, y_pred):
        y_pred_binary = self.binarize_predictions(predictions=y_pred)
        # False negatives: both y_true is 1 and y_pred is 0
        return np.sum((y_true == 1) & (y_pred_binary == 0))
    
class AUC_OneVRest(Metric):
    def __init__(self):
        super().__init__(name="AUC_OneVRest", maximize=True, is_multi_target=True)

    def calculate(self, y_true, y_pred):
        results = {}
        num_classes = y_pred.shape[1]  # Number of classes based on predictions
        auc_scores = []

        for class_idx in range(num_classes):
            # Create binary labels for the current class
            y_true_binary = (y_true == class_idx).astype(int)  # Current class as positive class

            # Compute AUC for the current class against all others
            try:
                auc_score = roc_auc_score(y_true_binary, y_pred[:, class_idx])
                results[f"{class_idx+1}_vs_Rest"] = auc_score
                auc_scores.append(auc_score)
            except ValueError:
                # Handle case with no valid samples for the class
                results[f"{class_idx+1}_vs_Rest"] = None

        # Compute average AUC excluding None values
        if auc_scores:
            results["avg"] = np.mean(auc_scores)
        else:
            results["avg"] = None

        return results
    

class AUC_OneVOne(Metric):
    def __init__(self):
        super().__init__(name="AUC_OneVOne", maximize=True, is_multi_target=True)

    def calculate(self, y_true, y_pred):
        results = {}
        auc_scores = []
        num_classes = y_pred.shape[1]  # Number of classes based on predictions
        class_combinations = list(combinations(range(num_classes), 2))

        for (class_a, class_b) in class_combinations:
            # Create mask for the current pair of classes
            mask = (y_true == class_a) | (y_true == class_b)
            
            # Apply mask to y_true and y_pred
            y_true_combined = y_true[mask]
            y_pred_combined = y_pred[mask.reshape(-1)][:, [class_a, class_b]]

            # Create binary labels for AUC computation
            y_true_binary = (y_true_combined == class_b).astype(int)  # Class_b as positive class

            # Normalize predictions for the current pair
            y_pred_combined = y_pred_combined / y_pred_combined.sum(axis=1, keepdims=True)

            # Compute AUC for the pair
            if y_pred_combined.shape[0] > 0:  # Ensure there are valid samples
                auc_score = roc_auc_score(y_true_binary, y_pred_combined[:, 1])
                results[f"{class_a}_vs_{class_b}"] = auc_score
                auc_scores.append(auc_score)
            else:
                results[f"{class_a}_vs_{class_b}"] = None  # Handle case with no valid samples

        # Compute average AUC excluding None values
        if auc_scores:
            results["avg"] = np.mean(auc_scores)
        else:
            results["avg"] = None


        return results

class MicroAUC(Metric):
    def __init__(self):
        super().__init__(name="MicroAUC", maximize=True, is_multi_target=True)
    
    def calculate(self, y_true, y_pred):
        # Convert y_true from index format to one-hot encoding
        if y_true.shape[-1] == 1:
            num_classes = y_pred.shape[1]
            y_true_one_hot = np.eye(num_classes)[y_true.reshape(-1)]
        else:
            y_true_one_hot = y_true
        
        # Compute micro-averaged AUC
        return roc_auc_score(y_true_one_hot, y_pred, average='micro', multi_class='ovr')

class MicroAccuracy(ThreasholdBasedMetric):
    def __init__(self, threshold: float = 0.5):
        super().__init__(name="MicroAccuracy", maximize=True, threshold=threshold, is_multi_target=True)
    
    def calculate(self, y_true, y_pred):
        """
        Compute micro-accuracy for multi-label classification.

        Micro-accuracy treats all label predictions across all samples
        as a single pool and computes overall accuracy.

        Args:
            y_true (np.ndarray): Ground truth labels (shape: [num_samples, num_classes] or [num_samples, 1]).
            y_pred (np.ndarray): Predicted logits/probabilities (shape: [num_samples, num_classes]).

        Returns:
            float: Micro-accuracy score.
        """
        # Binarize predictions using the threshold
        y_pred_binarized = self.binarize_predictions(y_pred)

        # Convert y_true to one-hot encoding if it's in index format
        if y_true.shape[-1] == 1:
            num_classes = y_pred.shape[1]
            y_true_one_hot = np.eye(num_classes)[y_true.reshape(-1)]
        else:
            y_true_one_hot = y_true

        # Flatten both arrays for micro-accuracy computation
        y_true_flat = y_true_one_hot.flatten()
        y_pred_flat = y_pred_binarized.flatten()

        # Compute and return micro-accuracy
        return accuracy_score(y_true_flat, y_pred_flat)


class MultiLabelSampleAccuracy(ThreasholdBasedMetric):
    def __init__(self, threshold: float = 0.5):
        super().__init__(name="MultiLabelSampleAccuracy", maximize=True, threshold=threshold, is_multi_target=True)

    def calculate(self, y_true, y_pred):
        """
        Compute sample-wise accuracy for multi-label classification.

        A sample is considered correct only if all its predicted labels
        exactly match the ground truth labels.

        Args:
            y_true (np.ndarray): Ground truth binary labels (shape: [num_samples, num_classes]).
            y_pred (np.ndarray): Predicted logits/probabilities (shape: [num_samples, num_classes]).

        Returns:
            float: Sample-wise accuracy (percentage of exactly matched samples).
        """
        # Binarize predictions using the threshold
        y_pred_binarized = self.binarize_predictions(y_pred)

        # Compare entire label sets for each sample
        exact_matches = np.all(y_pred_binarized == y_true, axis=1)

        # Compute accuracy as the proportion of samples with fully correct labels
        return np.mean(exact_matches)


class MultiClassConfusionMatrix(Metric):
    def __init__(self):
        super().__init__(name="MultiClassConfusionMatrix", maximize=False, is_multi_target=True)

    def calculate(self, y_true, y_pred):
        results = {}
        num_classes = len(np.unique(y_true))  # Number of classes based on ground truth

        # Convert logits to predicted class indices
        y_pred_class = np.expand_dims(np.argmax(y_pred, axis=1), -1)

        # Initialize the confusion matrix
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

        for true_class in range(num_classes):
            for predicted_class in range(num_classes):
                # Count instances where ground truth is `true_class` and prediction is `predicted_class`
                confusion_matrix[true_class, predicted_class] = np.sum(
                    (y_true == true_class) & (y_pred_class == predicted_class)
                )

        # Populate results with detailed confusion matrix data
        for true_class in range(num_classes):
            for predicted_class in range(num_classes):
                results[f"gt: {true_class} -> pred: {predicted_class}"] = confusion_matrix[true_class, predicted_class]

        return results

class MultiClassConfusionMatrixPercentage(Metric):
    def __init__(self):
        super().__init__(name="MultiClassConfusionMatrixPercentage", maximize=False, is_multi_target=True)

    def calculate(self, y_true, y_pred):
        results = {}
        num_classes = len(np.unique(y_true))  # Number of classes based on ground truth

        # Convert logits to predicted class indices
        y_pred_class = np.expand_dims(np.argmax(y_pred, axis=1), -1)

        # Initialize the confusion matrix
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

        for true_class in range(num_classes):
            for predicted_class in range(num_classes):
                # Count instances where ground truth is `true_class` and prediction is `predicted_class`
                confusion_matrix[true_class, predicted_class] = np.sum(
                    (y_true == true_class) & (y_pred_class == predicted_class)
                )

        # Convert to percentages
        row_sums = confusion_matrix.sum(axis=1, keepdims=True)
        confusion_matrix_percentage = confusion_matrix / row_sums * 100

        # Populate results with detailed confusion matrix percentage data
        for true_class in range(num_classes):
            for predicted_class in range(num_classes):
                results[f"gt: {true_class} -> pred: {predicted_class} (%)"] = confusion_matrix_percentage[true_class, predicted_class]

        return results

class FocalLoss(Metric):
    def __init__(self, alpha: float = 2, beta: float = 4):
        super().__init__(name="FocalLoss", maximize=False, is_multi_target=False)
        self.alpha = alpha
        self.beta = beta

    def calculate(self, y_true, y_pred):
        eps = 1e-8
        
        # Ensure predictions are clipped to avoid log(0)
        y_pred = np.clip(y_pred, eps, 1 - eps)
        
        # Masks of positive and negative samples
        positive_mask = y_true == 1
        negative_mask = ~positive_mask
        
        # Compute loss for positive samples
        loss_pos = np.log(y_pred[positive_mask] + eps) * (1 - y_pred[positive_mask]) ** self.alpha
        
        # Compute loss for negative samples  
        loss_neg = np.log(1 - y_pred[negative_mask] + eps) * y_pred[negative_mask] ** self.alpha * (1 - y_pred[negative_mask]) ** self.beta
        
        # Combine losses
        loss = np.concatenate([loss_pos, loss_neg])
        
        # Return mean loss (negative because we want to minimize)
        return -loss.mean()
        
class AUC_Contiguous_Target(Metric):
    def __init__(self):
        super().__init__(name="AUC_Contiguous_Target", maximize=True, is_multi_target=False)
    
    def calculate(self, y_true, y_pred):
        
        target_thresholds = [0.25, 0.5, 0.75, 1.0]
        results = {}
        
        for threshold in target_thresholds:
            y_true_binarized = (y_true >= threshold).astype(int)
            auc_score = roc_auc_score(y_true_binarized, y_pred)
            results[f"AUC_Contiguous_{threshold}"] = auc_score
        
        return results

METRIC_MAP = {

    # General
    "SampleSize": SampleSize(),

    # Regression Metrics
    "MSE": MSE(),

    # Binary Classification Metrics
    "AUC": AUC(),
    "BCE": BCE(),
    "Accuracy": Accuracy(),
    "BalancedAccuracy": BalancedAccuracy(),
    "Precision": Precision(),
    "Recall": Recall(),
    "F1": F1(),
    "FocalLoss": FocalLoss(),
    "AUC_Contiguous_Target": AUC_Contiguous_Target(),


    # Confusion Matrix Metrics
    "TP": TP(),
    "FP": FP(),
    "TN": TN(),
    "FN": FN(),
    "MultiClassConfusionMatrix": MultiClassConfusionMatrix(),
    "MultiClassConfusionMatrixPercentage": MultiClassConfusionMatrixPercentage(),
    

    # Multi-Class Classification Metrics
    "CE": CE(),
    "MicroAUC": MicroAUC(),
    "MicroAccuracy": MicroAccuracy(),
    "AUC_OneVOne": AUC_OneVOne(),
    "AUC_OneVRest": AUC_OneVRest(),
    "Top1Accuracy": Top1Accuracy(),
    "Top1BalancedAccuracy": Top1BalancedAccuracy(),

    # Multi-Label Metrics
    "MultiLabelSampleAccuracy": MultiLabelSampleAccuracy(),

}
    

class MetricManager:

    TABLE_COLUMNS = ['model_epoch_label', 'dataset', 'metric', 'target', 'value']

    def __init__(self, metrics_list: List[str], is_multi_class: bool = False):
        self.metrics = [METRIC_MAP[metric] for metric in metrics_list]
        self.metrics_table = None
        self.is_multi_class = is_multi_class

    def calculate_metrics(self, y_true, y_pred, dataset, model_epoch_label, display=False):

        # temporary table to store metrics
        table_entries = []
        
        # convert to numpy
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # clip if multi-class
        if self.is_multi_class:
            eps = 1e-8
            y_pred = np.clip(y_pred, eps, 1 - eps)
        
        if self.is_multi_class and y_pred.shape[1] > 2:
            y_true = np.expand_dims(y_true, axis=1)

        elif self.is_multi_class and y_pred.shape[1] == 2:
            y_pred = y_pred[:, [1]]
            y_true = np.reshape(y_true, (-1, 1))

        for metric in self.metrics:
            if metric.is_multi_target and y_pred.shape[1] > 1:
                # calculating metric value
                value = metric.calculate(y_true=y_true, y_pred=y_pred)
                
                table_entries.extend(
                    format_metric_table_entries(
                        target="all", metric=metric, value=value, model_epoch_label=model_epoch_label, dataset=dataset
                    )
                )
            else:
                for target_idx in range(y_true.shape[1]):
                    # calculating metric value
                    try:
                        value = metric.calculate(
                            y_true=y_true[:, target_idx], y_pred=y_pred[:, target_idx]
                        )
                    except:
                        value = None
                        
                    table_entries.extend(
                        format_metric_table_entries(
                            target=f"target_{target_idx + 1}",
                            metric=metric,
                            value=value,
                            model_epoch_label=model_epoch_label,
                            dataset=dataset
                        )
                    )

        tmp_metric_table = pd.DataFrame(table_entries)

        if display:
            print(f"\nModel {model_epoch_label} | {dataset} metrics: ")
            print(tabulate(tmp_metric_table, headers='keys', tablefmt='pretty', showindex=False))

        # appending the tmp table to global metrics table
        if self.metrics_table is not None: 
            self.metrics_table = pd.concat([self.metrics_table, tmp_metric_table], ignore_index=True)
        else:
            self.metrics_table = tmp_metric_table


    def get_best_epochs(self, dataset, metric):
        
        # filter for metric and dataset of interest 
        dataset_metrics = self.metrics_table[
            (self.metrics_table['dataset'] == dataset) &
            (self.metrics_table['metric'] == metric)
        ]
        
        best_epochs = {}
    
        # Loop through each unique target in the filtered dataset
        for target in dataset_metrics['target'].unique():
            target_metrics = dataset_metrics[dataset_metrics['target'] == target]
            
            # Find the epoch with the best metric value
            metric_obj = next(m for m in self.metrics if m.name == metric)
            
            # Handle case where all values are NaN (e.g., small sample size)
            if target_metrics['value'].isna().all() or target_metrics['value'].dropna().empty:
                # Default to the maximum epoch number (most trained)
                numeric_epochs = target_metrics[target_metrics['model_epoch_label'].apply(lambda x: isinstance(x, (int, float)))]
                if not numeric_epochs.empty:
                    best_row = numeric_epochs.iloc[-1]  # Last (highest) epoch
                else:
                    best_row = target_metrics.iloc[-1]
            else:
                if metric_obj.should_maximize:
                    best_idx = target_metrics['value'].idxmax()
                    if pd.isna(best_idx):
                        best_row = target_metrics.iloc[-1]
                    else:
                        best_row = target_metrics.loc[best_idx]
                else:
                    best_idx = target_metrics['value'].idxmin()
                    if pd.isna(best_idx):
                        best_row = target_metrics.iloc[-1]
                    else:
                        best_row = target_metrics.loc[best_idx]
            
            # Create a key for the target and store the best epoch
            best_epochs[target] = best_row['model_epoch_label']

        return best_epochs
    
    def summarize_best_epoch_metrics(self, dataset, metric, display) -> None:
        best_epochs_dict = self.get_best_epochs(dataset=dataset, metric=metric)
        for target, epoch in best_epochs_dict.items():
            summary_metrics = self.metrics_table[
                (self.metrics_table['dataset'] == dataset) &
                (self.metrics_table['model_epoch_label'] == epoch)
            ]
            model_label = f"target_{target}_best_{metric}"
            summary_metrics.loc[:, 'model_epoch_label'] = model_label
            
            # Only add EPOCH row if epoch is numeric
            if isinstance(epoch, (int, float)):
                table_entry = {
                    'model_epoch_label': model_label,
                    'dataset': dataset,
                    'metric': "EPOCH",
                    'target': target,
                    'value': float(epoch)
                }
                summary_metrics = pd.concat(
                    [summary_metrics, pd.DataFrame([table_entry])], ignore_index=True
                )

            self.metrics_table = pd.concat([self.metrics_table, summary_metrics], ignore_index=True)

            if display:
                print(f"\nModel {model_label} | {dataset} metrics: ")
                print(tabulate(summary_metrics, headers='keys', tablefmt='pretty', showindex=False))

    def save_metrics_table(self, save_path: str) -> None:
        if self.metrics_table is not None:
            self.metrics_table.to_csv(save_path, index=False)
        else:
            raise RuntimeError("Cannot save metrics table. No metrics have been recorded.")


def rank_prediction_quality(
    sample_keys: List[str],
    targets: List,
    predictions: List,
) -> Dict:
    
    # convert to numpy array
    targets = np.array(targets)
    predictions = np.array(predictions)
    
    n, d = targets.shape
    result = {}

    # Iterate over each target dimension (target_1, target_2, etc.)
    for target_idx in range(d):
        target_key = f"target_{target_idx}"
        result[target_key] = {}

        # Get the unique classes or values present in the target
        unique_classes = np.unique(targets[:, target_idx])

        # Process each class (for classification) or unique target value (for regression)
        for class_val in unique_classes:
            class_key = f"class_{int(class_val)}"
            result[target_key][class_key] = []

            # Find the indices of the samples that belong to this class/target value
            class_indices = np.where(targets[:, target_idx] == class_val)[0]

            # Get the corresponding predictions and targets for these indices
            class_predictions = predictions[class_indices, target_idx]
            class_targets = targets[class_indices, target_idx]
            class_sample_keys = [sample_keys[i] for i in class_indices]
            errors = np.abs(class_predictions - class_targets)
            quality_scores = -errors  # Lower error is better, so we negate for sorting

            # Pair sample keys with their corresponding quality score
            sample_quality_pairs = list(zip(class_sample_keys, quality_scores))

            # Sort based on quality score (higher score is better, thus reverse sorting)
            sorted_samples = sorted(sample_quality_pairs, key=lambda x: x[1], reverse=True)

            # Extract sorted sample keys
            result[target_key][class_key] = [sample[0] for sample in sorted_samples]

    return result


def format_metric_table_entries(target: str, metric: Metric, value: Union[float, dict], model_epoch_label: str, dataset: str) -> List[dict]:
    
    table_entries = []
    
    if isinstance(value, dict):
        for key, val in value.items():
            table_entry = {
                'model_epoch_label': str(model_epoch_label),
                'dataset': dataset,
                'metric': metric.name,
                'target': key,
                'value': val
            }
            table_entries.append(table_entry)
    else:
        table_entry = {
            'model_epoch_label': str(model_epoch_label),
            'dataset': dataset,
            'metric': metric.name,
            'target': target,
            'value': value
        }
        table_entries.append(table_entry)

    return table_entries