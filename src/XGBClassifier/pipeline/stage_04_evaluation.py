from XGBClassifier.config_mngr.configuration_manager import ConfigurationManager
from XGBClassifier.components.evaluation import Evaluation
from XGBClassifier import logger
from XGBClassifier.components.evaluation import Evaluation
import pickle


STAGE_NAME = "Evaluation stage"



class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        val_config = config.get_validation_config()
        xgb = pickle.load(open(val_config.model_save_filepath, 'rb'))
        evalu = Evaluation(config=val_config)

        evalu.model_evaluation(xgb , "one")
        
        evalu.precision_recall_trade_off(xgb)
        
        evalu.roc_auc(xgb)



if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main(xgb)
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
            