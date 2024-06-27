from src.XGBClassifier import logger
from src.XGBClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.XGBClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from src.XGBClassifier.pipeline.stage_03_training_cross_val import ModelTrainingPipeline
from src.XGBClassifier.pipeline.stage_04_evaluation import EvaluationPipeline
from src.XGBClassifier.pipeline.stage_05_mlflow import MLFlowTrackingPipeline

STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME = "Prepare base model"
try: 
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   prepare_base_model = PrepareBaseModelTrainingPipeline()
   xgb = prepare_base_model.main()
   
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e



STAGE_NAME = "Train and save model"
try:
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   obj = ModelTrainingPipeline()
   obj.main(xgb)
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
   logger.exception(e)
   raise e




STAGE_NAME = "Evaluation stage"
try:
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   model_evalution = EvaluationPipeline()
   model_evalution.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
   logger.exception(e)
   raise e



STAGE_NAME = "MLFlow stage"
try:
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   model_tracking = MLFlowTrackingPipeline()
   model_tracking.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
   logger.exception(e)
   raise e





'''
# Upload files to S3-
try:
    logger.info(f"*******************")
    logger.info(f">>>>>> stage Upload to S3 started <<<<<<")
    result = subprocess.run(['python', 'upload_to_s3.py'], capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Failed to upload files to S3: {result.stderr}")
        raise Exception(f"Failed to upload files to S3: {result.stderr}")
    else:
        logger.info(f"Successfully uploaded files to S3: {result.stdout}")
    logger.info(f">>>>>> stage Upload to S3 completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e
'''


