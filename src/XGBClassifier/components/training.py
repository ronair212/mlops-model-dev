import pickle
from XGBClassifier.entity.config_entity import TrainingConfig, HyperparameterOptimizationConfig
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from XGBClassifier import logger
from XGBClassifier.components.optimization import hyperparameter_optimization
import xgboost as xgb

class Training:
    def __init__(self, config: TrainingConfig, hpo_config: HyperparameterOptimizationConfig):
        self.config = config
        self.hpo_config = hpo_config

    def train_valid_generator(self):
        df = pd.read_csv(self.config.training_data, encoding='utf-8')
        features = df.columns.drop(['fraud'])
        target = 'fraud'

        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.config.test_size, random_state=self.config.random_state, stratify=y)
        return X_train, X_test, y_train, y_test

    def validate(self, model, X_train, y_train, score, n):
        '''this function is to validate the model across multiple stratified splits'''
        splits = StratifiedKFold(n_splits=n)
        validate = cross_val_score(model, X_train, y_train, scoring=score, cv=splits)
        logger.info(f"Cross Validation Scores: {validate}")
        logger.info(f"Scores Mean:  {validate.mean()}")
        logger.info(f"Scores Standard Deviation: {validate.std()}")

        model.fit(X_train, y_train)
        return model

    def train(self, xgb):
        X_train, X_test, y_train, y_test = self.train_valid_generator()
        logger.info(f"Model training initiated.")
        
        if self.hpo_config.use_hyperopt:
            logger.info(f"Hyperparameter optimization using Hyperopt initiated.")
            best_model, best_params, best_metrics = hyperparameter_optimization(X_train, y_train, X_test, y_test, self.hpo_config)
            model = best_model
        else:
            logger.info(f"Training without hyperparameter optimization.")
            model = self.validate( xgb  ,X_train,y_train, self.config.scoring ,self.config.cv)

        file_name = self.config.model_save_filepath

        # save model
        pickle.dump(model, open(file_name, "wb"))
        logger.info(f"Model saved in path {file_name}.")
        if self.hpo_config.use_hyperopt:
            logger.info(f"Best parameters: {best_params}")
            logger.info(f"Best metrics: {best_metrics}")
