from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    
    
@dataclass(frozen=True)
class PrepareBaseModelConfig:
    params_n_estimators: int
    params_learning_rate: float
    params_random_state: int
    params_subsample: float
    params_n_jobs: int
    params_max_depth: int
    
    
@dataclass(frozen=True)
class TrainingConfig:
    scoring: str
    cv: int
    training_data: Path
    model_save_filepath: Path
    stratify: str
    test_size: float
    random_state : int
    eval_results_folder: Path



@dataclass(frozen=True)
class EvaluationConfig:
    model_save_filepath: Path
    local_data_file: Path
    mlflow_uri: str
    all_params: dict
    eval_results_folder: Path
    mlflow_results_folder: Path
    
    
@dataclass(frozen=True)
class MLFlowConfig:
    model_save_filepath: Path
    experiment_name: str
    local_tracking_uri: str
    remote_tracking_uri: str
    all_params: dict
    eval_results_folder: Path
    mlflow_results_folder: Path


@dataclass(frozen=True)
class HyperparameterOptimizationConfig:
    max_evals: int
    params_space: dict
    random_state: int
    use_hyperopt: bool
