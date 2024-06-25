import mlflow

def get_or_create_experiment(experiment_name, tracking_uri):
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    try:
        experiment_id = client.create_experiment(experiment_name)
    except mlflow.exceptions.MlflowException:
        experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
    return experiment_id
