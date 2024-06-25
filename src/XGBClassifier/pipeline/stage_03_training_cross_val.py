from XGBClassifier.config_mngr.configuration_manager import ConfigurationManager
from XGBClassifier.components.training import Training
from XGBClassifier import logger


STAGE_NAME = "Training"



class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self , xgb):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        hpo_config = config.get_hyperparameter_optimization_config()
        
        training = Training(training_config, hpo_config)
        #print("completed till initializing configs")
        training.train(xgb)
        
    




if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
        


