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
       
        )
        
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        recall = recall_score(y_test, preds)
        
        return {'loss': -recall, 'status': STATUS_OK}
    '''
    param_space = {
        'n_estimators': hp.quniform('n_estimators', hpo_config.params_space['n_estimators']['min'], hpo_config.params_space['n_estimators']['max'], hpo_config.params_space['n_estimators']['step']),
        'max_depth': hp.quniform('max_depth', hpo_config.params_space['max_depth']['min'], hpo_config.params_space['max_depth']['max'], hpo_config.params_space['max_depth']['step']),
        'learning_rate': hp.uniform('learning_rate', hpo_config.params_space['learning_rate']['min'], hpo_config.params_space['learning_rate']['max']),
        'subsample': hp.uniform('subsample', hpo_config.params_space['subsample']['min'], hpo_config.params_space['subsample']['max']),
        'colsample_bytree': hp.uniform('colsample_bytree', hpo_config.params_space['colsample_bytree']['min'], hpo_config.params_space['colsample_bytree']['max']),
        'gamma': hp.uniform('gamma', hpo_config.params_space['gamma']['min'], hpo_config.params_space['gamma']['max'])
    }

    # Ensure param_space is a plain dictionary
    param_space = {k: v for k, v in param_space.items()}
    '''
    
    param_space = {
            'max_depth': hp.choice("max_depth", np.arange(1,20,1,dtype=int)),
            'gamma'    : hp.uniform("gamma", 0, 10e1),
            'seed' : 44
        }
    
    
    trials = Trials()
    
    best = fmin(fn=objective,
                space=param_space,
                algo=tpe.suggest,
                max_evals=500,
                trials=trials,
                )

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
