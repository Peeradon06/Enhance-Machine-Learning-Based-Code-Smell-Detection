import os
import sys
sys.path.append('../')

from numpy import mean
import joblib
from config.definitions import MODEL_DIR
from sklearn import metrics

class Classifier:
    """ 
    A class used to represent a Classifier

    ...

    Attributes
    ----------
    model : sklearn classifier class
        a model object from sklearn 
    model_name : str
        the name of the model
    training_time : float 
        the time used for training the model in second
    accuracy : float 
        the cross validation accuracy of the model 
    precision : float 
        the cross validation precion of the model
    recall : float 
        the cross validation recall of the model
    f1 : float 
        the cross validation f1 score of the model
    roc_auc : flot 
        the cross validation area under roc (roc_auc) of the model
    pred_accuracy : float 
        the predicton accuracy of the model 
    pred_precision : float 
        the prediction precion of the model
    pred_recall : float 
        the prediction recall of the model
    pred_f1 : float 
        the prediction f1 score of the model
    pred_roc_auc : flot 
        the prediction area under roc (roc_auc) of the model
    file_name : str
        the path name of the exported model in pkl format    
    """

    def __init__(self, model, model_name, scores=None) -> None:
        
        self.model = model
        self.model_name = model_name
        self.training_time = 0.0
        
        self.accuracy = 0.0
        self.precision = 0.0
        self.recall = 0.0
        self.f1 = 0.0
        self.roc_auc = 0.0
        
        self.pred_accuracy = 0.0
        self.pred_precision = 0.0
        self.pred_recall = 0.0
        self.pred_f1 = 0.0
        self.pred_roc = 0.0
        
        if scores:
            self.set_score(scores)

    def get_model(self):
        return self.model

    def set_trainning_time(self, time):
        self.training_time = time

    def set_score(self, scores):
        """ Set the model scores

        Parameters
        ----------
        scores : dict of float arrays of shape (n_splits,)
            Array of scores of the estimator for each run of the cross validation
        """

        self.scores = scores
        self.accuracy = mean(scores['test_accuracy'])
        self.precision = mean(scores['test_precision'])
        self.recall = mean(scores['test_recall'])
        self.f1 = mean(scores['test_f1'])
        self.roc_auc = mean(scores['test_roc_auc'])
    
    def set_prediction_score(self, X_test, y_test):
        """  Set the model prediction scores

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            The data to fit
        y_test : array-like of shape (n_samples,) 
            The target variable to  predict 
        """

        y_pred = self.model.predict(X_test)
 
        self.pred_accuracy = metrics.accuracy_score(y_test, y_pred)
        self.pred_precision = metrics.precision_score(y_test, y_pred, zero_division=1)
        self.pred_recall = metrics.recall_score(y_test, y_pred, zero_division=1)
        self.pred_f1 = metrics.f1_score(y_test, y_pred, zero_division=1)
        self.pred_roc = metrics.roc_auc_score(y_test, y_pred)
    
    def get_prediction_score(self):
        return {
            'pred_accuracy' : self.pred_accuracy,
            'pred_precision' : self.pred_precision,
            'pred_recall' : self.pred_recall,
            'pred_f1' : self.pred_f1,
            'pred_roc' : self.pred_roc
        }

    def export_model(self, sub_dir=None, pre_fix="") -> None:
        """ Export model into pkl file 

        Parameters
        ----------
        sub_dir : str
            A path of sub-directory to be exported 
        pre_fix : str
            A prefix of the model file name
        
        Example
        -------
        >>> export_model(sub_dir='baseline', prefix='baseline') 
        """

        self.file_name = pre_fix + "_" + self.model_name + ".pkl"

        try:
            if sub_dir:
                path = os.path.join(MODEL_DIR, sub_dir)
                file_path = os.path.join(MODEL_DIR, sub_dir, self.file_name)
            else:
                path = MODEL_DIR
                file_path = os.path.join(MODEL_DIR, self.file_name)

            # check if the path exist
            isdir = os.path.isdir(path)
            if not isdir:
                os.makedirs(path)
            joblib.dump(self.model, file_path)
            print(f'Model: {self.file_name} exported !')

        except Exception as e:
            print("Error while dumping the models : ", e)
    


