import hyperopt
from hyperopt import fmin, tpe, Trials, STATUS_OK, hp
import xgboost as xgb
from sklearn.metrics import recall_score
import numpy as np
from XGBClassifier import logger

def hyperparameter_optimization(X_train, y_train, X_test, y_test, hpo_config):
    def objective(params):
        model = xgb.XGBClassifier(
            max_depth=int(params['max_depth']),
            gamma=params['gamma'],
            n_estimators=int(params['n_estimators']),
            learning_rate=params['learning_rate'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            random_state=44
        )
        
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        recall = recall_score(y_test, preds)
        
        return {'loss': -recall, 'status': STATUS_OK}
    
    param_space = {
        'max_depth': hp.choice("max_depth", np.arange(1, 20, 1, dtype=int)),
        'gamma': hp.uniform("gamma", 0, 10),
        'n_estimators': hp.choice("n_estimators", np.arange(100, 1000, 10, dtype=int)),
        'learning_rate': hp.uniform("learning_rate", 0.01, 0.2),
        'subsample': hp.uniform("subsample", 0.5, 1),
        'colsample_bytree': hp.uniform("colsample_bytree", 0.5, 1)
    }
    
    trials = Trials()
    
    best = fmin(fn=objective,
                space=param_space,
                algo=tpe.suggest,
                max_evals=2,
                trials=trials)
    
    best_params = {
        'n_estimators': int(best['n_estimators']),
        'max_depth': int(best['max_depth']),
        'learning_rate': best['learning_rate'],
        'subsample': best['subsample'],
        'colsample_bytree': best['colsample_bytree'],
        'gamma': best['gamma']
    }

    logger.info(f"Best hyperparameters found: {best_params}")

    best_model = xgb.XGBClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        learning_rate=best_params['learning_rate'],
        subsample=best_params['subsample'],
        colsample_bytree=best_params['colsample_bytree'],
        gamma=best_params['gamma'],
        random_state=hpo_config.random_state,
        n_jobs=-1
    )

    best_model.fit(X_train, y_train)
    preds = best_model.predict(X_test)
    best_metrics = {
        'recall': recall_score(y_test, preds)
    }

    logger.info(f"Best model recall: {best_metrics['recall']}")

    return best_model, best_params, best_metrics