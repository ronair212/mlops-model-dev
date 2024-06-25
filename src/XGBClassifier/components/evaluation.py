from XGBClassifier.entity.config_entity import EvaluationConfig
from XGBClassifier import logger
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split , GridSearchCV ,StratifiedKFold , cross_val_score
import mlflow
import plotly.express as px
from sklearn.metrics import confusion_matrix , precision_recall_curve , roc_auc_score , roc_curve , classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import dagshub

from sklearn.metrics import classification_report, confusion_matrix
from XGBClassifier.utils.common import *
from pathlib import Path
import pandas as pd

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def train_valid_generator(self):
        df = pd.read_csv(self.config.local_data_file,encoding='utf-8')
        features = df.columns.drop(['fraud'])
        target = 'fraud'

        X = df[features]
        y = df[target]
        X_train , X_test ,y_train , y_test = train_test_split(X,y,test_size = 0.2 ,random_state =42 ,stratify=y)
        return X_train , X_test ,y_train , y_test 
        
        
    def precision_recall_trade_off(self, model):
    
        '''this function is to plot the precision-recall curve then
        printing the thresholds that achieves the highest recall'''
        X_train , X_test ,y_train , y_test  = self.train_valid_generator()
        y_proba = model.predict_proba(X_test)
        precision ,recall ,threshold = precision_recall_curve(y_test,y_proba[:,1])
        p_r_t = pd.DataFrame({'Threshold':threshold,'Precision':precision[:-1],'Recall':recall[:-1]})
        fig = px.line(
            p_r_t,
            x='Recall',
            y='Precision',
            title='Precision-Recall Curve',
            width=700,height=500,
            hover_data=['Threshold']
        )
        save_figure_with_timestamp(fig, prefix="precision_recall_curve")
        prt = p_r_t[(p_r_t['Recall'] == 1)].tail(10)

        logger.info(p_r_t[ (p_r_t['Recall']==1)].tail(10))
        prt_list = prt.to_dict(orient='records')
        prt_dict = {"data": prt_list}
        save_json(path=Path("prt.json"), data=prt_dict)
        
    
    def get_or_create_experiment_id(self, name):
        exp = mlflow.get_experiment_by_name(name)
        if exp is None:
            exp_id = mlflow.create_experiment(name)
            return exp_id
        return exp.experiment_id

    def model_evaluation(self, model, experiment_name, threshold=0.5):
        '''This function evaluates the model based on a given threshold
        and logs the evaluation metrics with both DagsHub and locally'''

        # Generate train and test data
        X_train, X_test, y_train, y_test = self.train_valid_generator()

        # Predict probabilities and binary predictions
        y_proba_test = model.predict_proba(X_test)
        y_pred_test = (y_proba_test[:, 1] >= threshold)

        # Print the classification report
        logger.info("\n", classification_report(y_test, y_pred_test, zero_division=0))

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred_test)
        precision = precision_score(y_test, y_pred_test)
        recall = recall_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test)

        # Metrics DataFrame
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Value': [accuracy, precision, recall, f1]
        })

        # Plot the evaluation metrics
        fig = px.line(metrics_df, x='Metric', y='Value', title='Evaluation Metrics')

        # Save the plot
        save_figure_with_timestamp(fig, prefix="evaluation_metrics")

        # Set up local MLflow tracking URI
        local_mlflow_dir = 'mlruns'
        mlflow.set_tracking_uri(local_mlflow_dir)

        # Get or create an experiment
        experiment_id = self.get_or_create_experiment_id(experiment_name)

        with mlflow.start_run(experiment_id=experiment_id):
            # Log model parameters
            model_params = self.config.all_params
            for param_name, param_value in model_params.items():
                mlflow.log_param(param_name, param_value)

            # Log evaluation metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

        # Log to DagsHub
        dagshub.init(repo_owner='ronair212', repo_name='mlops-model-dev', mlflow=True)
        #mlflow.set_tracking_uri('https://dagshub.com/ronair212/mlops-model-dev.mlflow')

        with mlflow.start_run(experiment_id=experiment_id):
            # Log model parameters
            for param_name, param_value in model_params.items():
                mlflow.log_param(param_name, param_value)

            # Log evaluation metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

        # Reset tracking URI to local after logging to DagsHub
        mlflow.set_tracking_uri(local_mlflow_dir)
        
        
    
    def roc_auc(self, model):
    
        '''this function plots the roc-auc curve and calculate the model ROC-AUC score '''
        X_train , X_test ,y_train , y_test  = self.train_valid_generator()
        y_proba = model.predict_proba(X_test)
        fpr ,tpr ,threshold = roc_curve(y_test,y_proba[:,1])
        fp_tp = pd.DataFrame({'Threshold':threshold,'FPR':fpr,'TPR':tpr})
        fig = px.line(
            fp_tp,
            x='FPR',
            y='TPR',
            title='ROC Curve',
            width=700,height=500,
            hover_data=['Threshold']
        )
        save_figure_with_timestamp(fig, prefix="roc-auc-curve")
        roc_auc_scores = roc_auc_score(y_test,y_proba[:,1])
        logger.info('Testing ROC-AUC Score: ',roc_auc_score(y_test,y_proba[:,1]))
        #save_json(path=Path("roc_auc_scores.json"), data=roc_auc_scores)
    
    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        mlflow.xgboost.autolog()
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )
            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
            else:
                mlflow.keras.log_model(self.model, "model")