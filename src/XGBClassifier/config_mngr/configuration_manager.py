from XGBClassifier.constants import *
import os
from pathlib import Path
from XGBClassifier.utils.common import read_yaml, create_directories
from XGBClassifier.entity.config_entity import (DataIngestionConfig, PrepareBaseModelConfig, TrainingConfig)




class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config


    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            params_n_estimators=self.params.n_estimators,
            params_random_state=self.params.random_state,
            params_n_jobs=self.params.n_jobs,
            params_max_depth=self.params.max_depth,
            params_learning_rate=self.params.learning_rate, 
            params_subsample=self.params.subsample, 
        )

        return prepare_base_model_config
    
    
    
    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        params = self.params
        training_data = training.local_data_file
        
        create_directories([Path(training.root_dir)])

        training_config = TrainingConfig(
            model_save_filepath=Path(training.model_save_filepath),
            training_data=Path(training_data),
            cv=params.cv,
            scoring=params.scoring,
        )
        return training_config