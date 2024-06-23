from XGBClassifier.entity.config_entity import EvaluationConfig
from XGBClassifier import logger
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split , GridSearchCV ,StratifiedKFold , cross_val_score
import mlflow
import plotly.express as px
from sklearn.metrics import confusion_matrix , precision_recall_curve , roc_auc_score , roc_curve , classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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
        
    
    def model_evaluation(self, model, color='Blues', threshold=0.5):
        '''This function is to evaluate the model based on a given threshold
        1--> print the classification report     
        2--> display and save the confusion matrix'''
        
        # Classification report
        X_train , X_test ,y_train , y_test  = self.train_valid_generator()
        y_proba_test = model.predict_proba(X_test)
        y_pred_test  = (y_proba_test[:,1] >= threshold)
        logger.info("\n", classification_report(y_test, y_pred_test, zero_division=0))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_test)
        cm_df = pd.DataFrame(cm, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])
        
        fig = px.imshow(cm_df, text_auto=True, color_continuous_scale=color, 
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        title="Confusion Matrix")
        
        # Save the confusion matrix plot using the utility function
        save_figure_with_timestamp(fig, prefix="confusion_matrix")
        
        '''
        # Calculate additional metrics
        accuracy = accuracy_score(y_test, y_pred_test)
        precision = precision_score(y_test, y_pred_test)
        recall = recall_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test)

        # Set up DagsHub repository URL
        dagshub_repo_url = "https://dagshub.com/<username>/<repository>.mlflow"
        mlflow.set_tracking_uri(dagshub_repo_url)

        # Example model parameters
        model_params = self.config.all_params

        # Log the heatmap and metrics with MLflow
        with mlflow.start_run():
            # Log the heatmap image
            mlflow.log_artifact(heatmap_path)
            
            # Log the confusion matrix as a parameter
            mlflow.log_param("confusion_matrix", cm.tolist())
            
            # Log model parameters
            for param_name, param_value in model_params.items():
                mlflow.log_param(param_name, param_value)
            
            # Log model metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            # Log the model itself
            # Assuming you have a trained model object named 'model'
            # Example for a scikit-learn model:
            # mlflow.sklearn.log_model(model, "model")
        '''
    
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
        save_json(path=Path("roc_auc_scores.json"), data=roc_auc_scores)
    
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