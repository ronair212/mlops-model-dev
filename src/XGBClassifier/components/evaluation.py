from XGBClassifier.entity.config_entity import EvaluationConfig
from XGBClassifier import logger
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from XGBClassifier.utils.common import save_figure_with_timestamp
from pathlib import Path


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def train_valid_generator(self):
        df = pd.read_csv(self.config.training_data,encoding='utf-8')
        features = df.columns.drop(['fraud'])
        target = 'fraud'

        X = df[features]
        y = df[target]
        X_train , X_test ,y_train , y_test = train_test_split(X,y,test_size = self.config.test_size ,random_state =self.config.random_state  ,stratify=y)
        return X_train , X_test ,y_train , y_test 
        
        
    def precision_recall_trade_off(self, model,X_test,y_test):
    
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
        prt = p_r_t[ (p_r_t['Recall']==1)].tail(10)
        logger.info(p_r_t[ (p_r_t['Recall']==1)].tail(10))
        save_json(path=Path("prt.json"), data=prt)
        
    
    def model_evaluation(self, model, X_test, y_test, color='Blues', threshold=0.5):
        '''This function is to evaluate the model based on a given threshold
        1--> print the classification report     
        2--> display and save the confusion matrix'''
        
        # Classification report
        y_proba_test = model.predict_proba(X_test)
        y_pred_test  = (y_proba_test[:,1] >= threshold)
        logger.info(classification_report(y_test, y_pred_test, zero_division=0))
        
        # Confusion matrix
        plt.figure(figsize=(5,4))
        sns.heatmap(confusion_matrix(y_test, y_pred_test), cmap=color, annot=True)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        # Save the confusion matrix plot using the utility function
        save_figure_with_timestamp(prefix="confusion_matrix")
    
    def roc_auc(model,X_test,y_test):
    
        '''this function plots the roc-auc curve and calculate the model ROC-AUC score '''
        
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
    
    