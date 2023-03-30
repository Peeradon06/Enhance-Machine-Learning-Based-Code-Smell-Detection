import sys
sys.path.append('../')
import time

from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from packages.classifier import Classifier


n_job = -1      # enalble parallel computing  

def train_optimize(X, y, conf, smell="", hpo="", n_splits=10, n_repeats=10, show_result=False, show_process=True, export_models=True):
    """ Train model using otimized hyper-parameter 

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The data to fit. Can be for example a list, or an array.
    y : array-like of shape (n_samples,)
        The target variable to try to predict
    conf : dict
        Dictionary contains hyper-parameter configuration of [DT, RF]
    smell : str
        The name of code smell
    hpo : str
        The name of hyper-parameter optimization method 
    n_split : int 
        Number of folds of the KFold
    n_repeats : int
        Number of times cross-validator needs to be repeated
    show_result : bool 
        The toggle for printing out the result recall value of DT and RF
    show_process : bool
        The toggle for printing out the process
    export_models : bool
        The toggle for exporting the model.pkl file
    
    Return 
    ------
    models : list 
        List contains Classfiers objects [decision_tree, random_forest]
    
    Example
    -------
    >>> bo_models[smell] = train_optimize(X_train, y_train, conf=bo_best[smell], smell="Data class", hpo="bo", n_splits=10, n_repeats=10)
    """

    if show_process:
        print(f'creating model using hyper-params from {hpo.upper()} . . .')
    
    scoring = {
        'accuracy' : make_scorer(accuracy_score),
        'precision' : make_scorer(precision_score, zero_division=1),
        'recall' : make_scorer(recall_score, zero_division=1),
        'f1' : make_scorer(f1_score, zero_division=1),
        'roc_auc' : make_scorer(roc_auc_score)
    }
    
    dt_params, rf_params = conf
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)
    
    # decision tree
    st = time.time() # time counter

    d_clf = DecisionTreeClassifier(**dt_params)
    d_clf.fit(X, y)
    dt_scores = cross_validate(d_clf, X, y, cv=cv, scoring=scoring, n_jobs=n_job)
    
    et = time.time()
    dt_elapsed_time = et - st

    # Random forest
    st = time.time() # time counter

    f_clf = RandomForestClassifier(**rf_params)
    f_clf.fit(X, y)
    rf_scores = cross_validate(f_clf, X, y, cv=cv, scoring=scoring, n_jobs=n_job)

    et = time.time()
    rf_elapsed_time = et - st
    
    # store to Classifier objects
    dt_model_name = smell+"_"+"decision_tree"
    dt_clf = Classifier(d_clf, dt_model_name, dt_scores)
    dt_clf.set_trainning_time(dt_elapsed_time)

    rf_model_name = smell+"_"+"random_forest"
    rf_clf = Classifier(f_clf, rf_model_name, rf_scores)
    rf_clf.set_trainning_time(rf_elapsed_time)

    # export models
    if export_models:
        dt_clf.export_model(sub_dir=hpo, pre_fix=hpo)
        rf_clf.export_model(sub_dir=hpo, pre_fix=hpo)
    
    if show_process:
        print("Done ! ")

    if show_result:
        print(f'[{hpo.upper()} - Decision tree] recall: {dt_clf.recall:.5f}')
        print(f'[{hpo.upper()} - Random forest] recall: {rf_clf.recall:.5f}')

    models = [dt_clf, rf_clf]

    return models

def train_baseline(X, y, smell="", n_splits=10, n_repeats=10, show_result=False, show_process=True, export_models=True, prefix="baseline"):
    """ Train baseline model

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The data to fit. Can be for example a list, or an array.
    y : array-like of shape (n_samples,)
        The target variable to try to predict
    n_split : int 
        Number of folds of the KFold
    n_repeats : int
        Number of times cross-validator needs to be repeated
    show_result : bool 
        The toggle for printing out the result recall value of DT and RF
    show_process : bool
        The toggle for printing out the process
    export_models : bool
        The toggle for exporting the model.pkl file
    prefix : str
        The prefix name of model.pkl files

    Return 
    ------
    models : list 
        List contains Classfiers objects [decision_tree, random_forest]
    
    Example
    -------
    >>> baseline_models[smell] = train_baseline(X_train, y_train, smell=smell, n_splits=10, n_repeats=10)
    """
    
    if show_process:
        print("creating baseline . . .")

    # define performance metrices
    # scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    scoring = {
        'accuracy' : make_scorer(accuracy_score),
        'precision' : make_scorer(precision_score, zero_division=1),
        'recall' : make_scorer(recall_score, zero_division=1),
        'f1' : make_scorer(f1_score, zero_division=1),
        'roc_auc' : make_scorer(roc_auc_score)
    }

    # define cross validation settings
    n_splits = n_splits
    n_repeats = n_repeats
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)

    # Decision tree
    st = time.time()

    d_clf = DecisionTreeClassifier()
    d_clf.fit(X, y)
    dt_scores = cross_validate(d_clf, X, y, cv=cv, scoring=scoring, n_jobs=n_job)

    et = time.time()
    dt_elapsed_time = et - st

    # Random forest
    st = time.time()

    f_clf = RandomForestClassifier()
    f_clf.fit(X, y)
    rf_scores = cross_validate(f_clf, X, y, cv=cv, scoring=scoring, n_jobs=n_job)

    et = time.time()
    rf_elasped_time = et - st


    # store to Classifier objects
    dt_model_name = smell+"_"+"decision_tree"
    dt_clf = Classifier(d_clf, dt_model_name, dt_scores)
    dt_clf.set_trainning_time(dt_elapsed_time)

    rf_model_name = smell+"_"+"random_forest"
    rf_clf = Classifier(f_clf, rf_model_name, rf_scores)
    rf_clf.set_trainning_time(rf_elasped_time)

    # export models
    if export_models:
        dt_clf.export_model(sub_dir="baseline", pre_fix=prefix)
        rf_clf.export_model(sub_dir="baseline", pre_fix=prefix)

    if show_process:
        print("Done ! ")

    if show_result:
        print(f'[Baseline - Decision tree] recall: {dt_clf.recall:.5f}')
        print(f'[Baseline - Random forest] recall: {rf_clf.recall:.5f}')

    models = [dt_clf, rf_clf]

    return models