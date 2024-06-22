import os
from pathlib import Path
from XGBClassifier.entity.config_entity import PrepareBaseModelConfig
from xgboost import XGBClassifier, plot_importance
from XGBClassifier import logger

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        
        
    def prepare_XGBClassifier_model(self):
        xgb = XGBClassifier(n_estimators=self.config.params_n_estimators ,
                            random_state=self.config.params_random_state ,
                            n_jobs= self.config.params_n_jobs ,)
        
        logger.info(f"Initialized XGBClassifier with following parameters {self.config}")  
        return xgb