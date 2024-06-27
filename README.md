# MLOPS Model development


## Workflows

### STEPS for model training:

Clone the repository

```bash
https://github.com/ronair212/mlops-model-dev
```

### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n fraud_env python=3.8 -y
```

```bash
conda activate fraud_env
```


### STEP 02- Install the requirements
```bash
pip install -r requirements.txt
```

### STEP 03- Set environment variables
To set up your environment for tracking with MLflow, you need to export the following environment variables:

```bash
export MLFLOW_TRACKING_URI=<your_tracking_uri>
export MLFLOW_TRACKING_USERNAME=<your_username>
export MLFLOW_TRACKING_PASSWORD=<your_password>
export PYTHONPATH=$PYTHONPATH:<your_project_src_path>
```

### STEP 03- Modify parameters as required for hyperparameter tuning

```bash
#  Run the following command
vi params.yaml
```


### STEP 03- Run the following command to start model training

```bash

python main.py
```


### STEP 03- Verify results 

```bash

cd /artifacts/results/
```





## MLflow



##### cmd
- mlflow ui

### dagshub
[dagshub](https://dagshub.com/)


## About MLflow & DVC

MLflow

 - Its Production Grade
 - Trace all of your expriements
 - Logging & taging your model





# ECR uri 
654654149355.dkr.ecr.us-east-1.amazonaws.com/mlops-dev