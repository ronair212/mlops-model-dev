import mlflow
import mlflow.xgboost
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import json
from XGBClassifier.entity.config_entity import MLFlowConfig




class Tracking:
    def __init__(self, config: MLFlowConfig):
        self.config = config
    
    def log_mlflow(self, local_experiment_id, remote_experiment_id, config, X_train, X_test, y_train, y_test, model, local_tracking_uri, remote_tracking_uri):
        mlflow.set_tracking_uri(local_tracking_uri)

        with mlflow.start_run(experiment_id=local_experiment_id) as run:
            run_id = run.info.run_id
            mlflow.log_params(config)

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            cm = confusion_matrix(y_test, y_pred)

            mlflow.log_metric('accuracy', acc)
            mlflow.log_metric('precision', precision)
            mlflow.log_metric('recall', recall)
            
            with open('confusion_matrix.json', 'w') as f:
                json.dump(cm.tolist(), f)
            mlflow.log_artifact('confusion_matrix.json')

            mlflow.xgboost.log_model(model, 'xgboost-model')

        mlflow.set_tracking_uri(remote_tracking_uri)
        with mlflow.start_run(experiment_id=remote_experiment_id) as run:
            run_id = run.info.run_id
            mlflow.log_params(config)
            mlflow.log_metric('accuracy', acc)
            mlflow.log_metric('precision', precision)
            mlflow.log_metric('recall', recall)
            mlflow.log_artifact('confusion_matrix.json')
            mlflow.xgboost.log_model(model, 'xgboost-model')
