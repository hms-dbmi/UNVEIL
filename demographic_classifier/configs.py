from pydantic import BaseModel, field_validator, model_validator
from pathlib import Path
from typing import Dict, List

class TrainingConfigs(BaseModel):
    
    # where to save the results of the experiment
    save_dir: str

    # data type
    is_multi_class: bool = False

    # file paths
    train_targets_file_path_list: List[str]
    val_test_targets_file_path_list: List[str]
    features_dir_path_list: List[str]

    # model arguments
    model_type: str
    model_init_args: Dict

    # metrics arguments
    metrics: List[str]
    best_model_selector_metrics_list: List[str]

    # dataset arguments
    tr_dataset_type: str
    tr_dataset_init_args: Dict
    collate_fn: str | None = None
    class_balance: bool = False
    sample_names_balance: List[str] = []
    val_test_dataset_type: str | None = None
    val_test_dataset_init_args: Dict | None = None

    # training related arguments
    optimizer_type: str
    optimizer_init_args: Dict
    loss_function: str
    loss_function_init_args: Dict = {}
    n_epochs: int = 100
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    cross_validation_folds: int | None = None
    early_stopping_patience: int | None = None

    # data splitting
    sample_identification_regex: str = ".+"
    val_from_train_ratio: float = 0.1               # when using cross validation
    validation_size: float = 0.1                    # when NOT using cross validation
    test_size: float = 0.1                          # when NOT using cross validation
    train_val_test_split_path: str | None = None    # when NOT using cross validation

    # external validation data paths
    ex_val_features_dir_path_list: List[str] = []
    ex_val_targets_file_path_list: List[str] = []
    ex_val_dataset_type: str | None = None          # if none, default to training setup
    ex_val_dataset_init_args: Dict | None = None    # if none, default to training setup

    # number of retries
    n_retries: int = 0


    class Config:
        # Setting protected_namespaces to an empty tuple to avoid the warning
        protected_namespaces: tuple = ()

    # Validator to ensure file paths are valid POSIX paths
    @field_validator('train_targets_file_path_list', 'val_test_targets_file_path_list', 'features_dir_path_list')
    def validate_posix_path(cls, v):
        if v is not None:
            for path in v:
                path = Path(path)
                # Check if it's a valid POSIX path (absolute and no drive for POSIX)
                if not path.is_absolute() or path.drive:  
                    raise ValueError(f'{path} is not a valid POSIX path')
        return v
    
    # Validator to ensure validation_size and test_size are above 0 and their sum is less than 1
    @field_validator('validation_size', 'test_size', mode='before')
    def validate_sizes(cls, v, field):
        # Ensure size is above 0
        if v <= 0:
            raise ValueError(f"{field.name} must be greater than 0")
        
        return v
    
    # Model validator to ensure val_test_targets_file_path_list is in train_targets_file_path_list
    @model_validator(mode='after')
    def validate_val_test_in_train(cls, values):
        val_test_targets_file_path_list = values.val_test_targets_file_path_list
        train_targets_file_path_list = values.train_targets_file_path_list

        for path in val_test_targets_file_path_list:
            if path not in train_targets_file_path_list:
                raise ValueError(f"{path} must be included in train_targets_file_path_list")
        
        return values
    
    # Model validator to ensure that all best_model_selector_metrics are in metrics list
    @field_validator('best_model_selector_metrics_list')
    def validate_metrics(cls, v, info):
        metrics = info.data.get('metrics', [])
        for selector_metric in v:
            if selector_metric not in metrics:
                raise ValueError(
                    f"Selector metric {selector_metric} must be included in the metrics list"
                )
        return v


class InferenceConfigs(BaseModel):
    
    # where to save the results of the experiment
    save_dir: str

    # location of the model bundle
    pt_bundle_path: str | List[str]

    # data type
    is_multi_class: bool = False

    # file paths
    targets_file_path_list: List[str]
    features_dir_path_list: List[str]

    # metrics arguments
    metrics: List[str]
    
    # dataset arguments
    dataset_type: str
    dataset_init_args: Dict

    # which data to run inference for
    data_split_path: str | None = None
    data_split_name: str | None = None

    # intermediate layers to capture output
    intermediate_layers_hooks: List[str] = []

    class Config:
        # Setting protected_namespaces to an empty tuple to avoid the warning
        protected_namespaces: tuple = ()