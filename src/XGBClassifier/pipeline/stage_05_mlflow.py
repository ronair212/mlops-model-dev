from XGBClassifier.config_mngr.configuration_manager import ConfigurationManager
from XGBClassifier.components.evaluation import Evaluation
from XGBClassifier import logger
from XGBClassifier.utils.experiment import *
import mlflow
from XGBClassifier.components.tracking import Tracking
import pickle

STAGE_NAME = "MLFlow stage"

class MLFlowTrackingPipeline:
    def __init__(self):
        self.config = ConfigurationManager()
        self.evaluation = Evaluation(self.config.get_validation_config())

    def main(self):
        mlflow_config = self.config.get_mlflow_config()
        
        local_experiment_id = get_or_create_experiment(mlflow_config.experiment_name, mlflow_config.local_tracking_uri)
        remote_experiment_id = get_or_create_experiment(mlflow_config.experiment_name, mlflow_config.remote_tracking_uri)
        X_train, X_test, y_train, y_test = self.evaluation.train_valid_generator()

        tracking = Tracking(config=mlflow_config)
        xgb = pickle.load(open(mlflow_config.model_save_filepath, 'rb'))
        
        tracking.log_mlflow(local_experiment_id, remote_experiment_id, mlflow_config.all_params , X_train, X_test, y_train, y_test, xgb, mlflow_config.local_tracking_uri, mlflow_config.remote_tracking_uri)

if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = MLFlowTrackingPipeline()
        obj.main(xgb)
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
