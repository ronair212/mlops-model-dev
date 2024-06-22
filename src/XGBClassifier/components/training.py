import pickle
from XGBClassifier.entity.config_entity import TrainingConfig
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split , GridSearchCV ,StratifiedKFold , cross_val_score
from XGBClassifier import logger


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    
    def train_valid_generator(self):
        df = pd.read_csv(self.config.training_data,encoding='utf-8')
        features = df.columns.drop(['fraud'])
        target = 'fraud'

        X = df[features]
        y = df[target]
        X_train , X_test ,y_train , y_test = train_test_split(X,y,test_size = 0.2 ,random_state = 42 ,stratify=y)
        return X_train , X_test ,y_train , y_test 
        
        
    def validate(self, model,X_train,y_train,score,n):
    
        '''this function is to validate the model across multiple stratified splits'''
        
        splits = StratifiedKFold(n_splits=n)
        validate = cross_val_score(model,X_train,y_train,scoring=score,cv=splits)
        logger.info(f"Cross Validation Scores: {validate}")
        logger.info(f"Scores Mean:  {validate.mean()}")
        logger.info(f"Scores Standard Deviation: {validate.std()}")
        
        model.fit(X_train,y_train)
        return model


    def train(self, xgb):
        X_train , X_test ,y_train , y_test  = self.train_valid_generator()
        logger.info(f"Model training and cross validation initiated.")
        model = self.validate( xgb  ,X_train,y_train, self.config.scoring ,self.config.cv)
        
        file_name = self.config.model_save_filepath

        # save model
        pickle.dump(model, open(file_name, "wb"))
        logger.info(f"Model saved in path {file_name}.")
