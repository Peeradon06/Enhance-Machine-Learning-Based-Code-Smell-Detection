import sys
sys.path.append('../')
import time
import numpy as np
from numpy import mean
import pandas as pd
from sklearn.model_selection import cross_val_score,  StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from hyperopt import fmin, tpe, hp
import optunity
import optunity.metrics
from smac.configspace import ConfigurationSpace
from smac.scenario.scenario import Scenario
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter
from smac.facade.smac_hpo_facade import SMAC4HPO


''' 
    - Functions for optimize hyperparameter using 
        - bo_tpe via hyperopt
        - pso via optunity
        - smac via SMAC
    - Contains: 2 ML models -> decision tree and random forest 
    - Objective function = minimize the value of recall (inverted) -> -recall : lower = better
    - Return: 
        - list of best hyperparameters configuration of each ML models: 
        [decision_tree, random_forest]
        - list of computational time : [decision_tree_ct, random_forest_ct]
    '''

# ------------------------- HPO setup -------------------------
fold = 10   
max_evals = 50   
show_result = False

def float_hypers_to_int(dt_best, rf_best):
    """ Turn float hyper-parameter to int 
    
    Parameters
    ----------
    dt_best : dict 
        Dictionary contains hyper-parameter configuration for DT model 
    rf_best : dict 
        Dictionary contains hyper-parameter configuration for RF model
    
    Return
    ------
    new_dt_best : dict 
        Dictionary contains hyper-parameter configuration without float type for DT model
    new_rf_best : dict 
        Dictionary contains hyper-parameter configuration without float type for RF model 
    
    Example 
    -------
    >>> dt_best, rf_best = float_hypers_to_int(dt_best, rf_best)
    """

    new_dt_best = {k: int(v) if type(v) == float else v for(k, v) in dt_best.items()}
    new_rf_best = {k: int(v) if type(v) == float else v for(k, v) in rf_best.items()}
    
    return new_dt_best, new_rf_best

def smac(X, y, num_folds=fold, max_evals=max_evals, show_result=show_result):
    """ Optimize using SMAC via SMAC library 

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The data to fit. Can be for example a list, or an array.
    y : array-like of shape (n_samples,)
        The target variable to try to predict
    num_folds : int 
        Number of folds of the KFold 
    max_evals : int 
        Number of max evaluation function of the optimization method
    show_process : bool
        The toggle for printing out the process
    
    Return 
    ------
    best_config : list 
        List of best hyperparameters configuration of DT and RF models 
    compute_time : list 
        List of of computational time of DT and RF in second 

    Example 
    -------
    >>> smac_best[smell], smac_ct[smell] = smac(X_train, y_train, max_evals=50)
    """

    print("Optmize using SMAC . . . ", end=" ")
    
    # define hyper-parameters
    criterion = CategoricalHyperparameter(
        "criterion", ["gini", "entropy"], default_value="gini")
    max_depth = UniformIntegerHyperparameter(
        "max_depth", 5, 50, default_value=5)
    min_samples_split = UniformIntegerHyperparameter(
        "min_samples_split", 2, 11, default_value=2)
    min_samples_leaf = UniformIntegerHyperparameter(
        "min_samples_leaf", 1, 11, default_value=1)
    max_features = UniformIntegerHyperparameter(
        "max_features", 1, 64, default_value=1)
    n_estimators = UniformIntegerHyperparameter(
        "n_estimators", 100, 300, default_value=100)

    # ----------------- decision tree -----------------
    st = time.time()   

    # define objective functions
    def dt_objective(params):
        clf = DecisionTreeClassifier(**params)
        score = mean(cross_val_score(clf, X, y, scoring='recall', cv=StratifiedKFold(
            n_splits=num_folds), n_jobs=-1))
        return -score

    # define search spaces
    dt_space = ConfigurationSpace()
    dt_space.add_hyperparameter(criterion)
    dt_space.add_hyperparameter(max_depth)
    dt_space.add_hyperparameter(min_samples_split)
    dt_space.add_hyperparameter(min_samples_leaf)
    dt_space.add_hyperparameter(max_features)

    # create scenario objects
    dt_scenario = Scenario(
        {
            "run_obj": "quality",  # optimize quality
            "runcount_limit": max_evals,  # max. number of evaluations
            "cs": dt_space,  # configuration space
            "deterministic": True
        }
    )

    # optimize
    dt_smac = SMAC4HPO(scenario=dt_scenario,
                       rng=np.random.RandomState(42), tae_runner=dt_objective)
    dt_best = dt_smac.optimize()

    et = time.time()
    dt_ct = et - st

    # ----------------- random forest -----------------
    st = time.time()

    # define objective functions
    def rf_objective(params):
        clf = RandomForestClassifier(**params)
        score = mean(cross_val_score(clf, X, y, scoring='recall', cv=StratifiedKFold(
            n_splits=num_folds), n_jobs=-1))
        return -score

    # define search spaces
    rf_space = ConfigurationSpace()
    rf_space.add_hyperparameter(criterion)
    rf_space.add_hyperparameter(max_depth)
    rf_space.add_hyperparameter(min_samples_split)
    rf_space.add_hyperparameter(min_samples_leaf)
    rf_space.add_hyperparameter(max_features)
    rf_space.add_hyperparameter(n_estimators)

    # create scenario objects
    rf_scenario = Scenario(
        {
            "run_obj": "quality",           # optimize quality
            "runcount_limit": max_evals,    # max. number of evaluations
            "cs": rf_space,                 # configuration space
            "deterministic": True
        }
    )
    
    # optimize
    rf_smac = SMAC4HPO(scenario=rf_scenario,
                       rng=np.random.RandomState(42), tae_runner=rf_objective)
    rf_best = rf_smac.optimize()

    et = time.time()
    rf_ct = et - st

    print("Done !")

    if show_result:
        inc_value = dt_objective(dt_best)
        print("Optimized Value: %.2f" % (inc_value))
        print(dt_best, "\n")

        inc_value = dt_objective(rf_best)
        print("Optimized Value: %.2f" % (inc_value))
        print(rf_best, "\n")
    

    dt_best, rf_best = float_hypers_to_int(dt_best, rf_best)
    
    best_config = [dt_best, rf_best]
    compute_time = [dt_ct, rf_ct]

    return best_config, compute_time

def pso(X, y, num_folds=fold, max_evals=max_evals, show_result=show_result):
    """ Optimize using Particle Swarm Optimization (PSO) via Optunity

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The data to fit. Can be for example a list, or an array.
    y : array-like of shape (n_samples,)
        The target variable to try to predict
    num_folds : int 
        Number of folds of the KFold 
    max_evals : int 
        Number of max evaluation function of the optimization method
    show_process : bool
        The toggle for printing out the process
    
    Return 
    ------
    best_config : list 
        List of best hyperparameters configuration of DT and RF models 
    compute_time : list 
        List of of computational time of DT and RF in second 

    Example 
    -------
    >>> pso_best[smell], pso_ct[smell] = pso(X_train, y_train, max_evals=50)
    """

    print("Optmize using PSO . . . ", end=" ")

    data = X.to_numpy()
    labels = y.tolist()

    # ----------------- decision tree -----------------
    st = time.time()

    # define configuration space
    dt_space = {
        'criterion': [0, 1],
        'max_depth': [5, 50],
        'min_samples_split': [2, 11],
        'min_samples_leaf': [1, 11],
        'max_features': [1, 64]
    }

    # define objective function
    @optunity.cross_validated(x=data, y=labels, num_folds=num_folds)
    def dt_objective(x_train, y_train, x_test, y_test, criterion=None, max_depth=None, min_samples_split=None, min_samples_leaf=None, max_features=None):
        # fit the models
        if criterion < 0.5:
            cri = 'gini'
        else:
            cri = 'entropy'

        clf = DecisionTreeClassifier(criterion=cri,
                                     max_depth=int(max_depth),
                                     min_samples_leaf=int(min_samples_leaf),
                                     min_samples_split=int(min_samples_split),
                                     max_features=int(max_features))
        score = mean(cross_val_score(clf, X, y, scoring='recall', cv=StratifiedKFold(
            n_splits=num_folds), n_jobs=-1))

        return -score

    # optimize
    dt_best, dt_info, _ = optunity.minimize(dt_objective,
                                            solver_name='particle swarm',
                                            num_evals=max_evals,
                                            **dt_space)

    et = time.time()
    dt_ct = et - st 


    # ----------------- random forest -----------------
    st = time.time()

    # define configuration space
    rf_space = {
        'criterion': [0, 1],
        'max_depth': [5, 50],
        'min_samples_split': [2, 11],
        'min_samples_leaf': [1, 11],
        'max_features': [1, 64],
        'n_estimators': [100, 300]
    }

    # define objective function
    @optunity.cross_validated(x=data, y=labels, num_folds=num_folds)
    def rf_objective(x_train, y_train, x_test, y_test, criterion=None, max_depth=None, min_samples_split=None, min_samples_leaf=None, max_features=None, n_estimators=None):
        # fit the models
        if criterion < 0.5:
            cri = 'gini'
        else:
            cri = 'entropy'

        clf = RandomForestClassifier(criterion=cri,
                                     max_depth=int(max_depth),
                                     min_samples_leaf=int(min_samples_leaf),
                                     min_samples_split=int(min_samples_split),
                                     max_features=int(max_features),
                                     n_estimators=int(n_estimators))
        score = mean(cross_val_score(clf, X, y, scoring='recall', cv=StratifiedKFold(
            n_splits=num_folds), n_jobs=-1))

        return -score

    # optimize
    rf_best, rf_info, _ = optunity.minimize(rf_objective,
                                            solver_name='particle swarm',
                                            num_evals=max_evals,
                                            **rf_space)
    
    et = time.time()
    rf_ct = et - st

    print("Done !\n")

    if show_result:
        print(f'--- Desicion tree best: {dt_best}')
        print(f'--- Decsion tree best recall: {dt_info.optimum}')
        print(f'=== Random forest best: {rf_best}')
        print(f'=== Random forest best recall: {rf_info.optimum}')
    
    def map_criterion_pso(conf):
        if conf < 0.5:
            cri = 'gini'
        else:
            cri = 'entropy'
        return cri
    
    dt_best['criterion'] = map_criterion_pso(dt_best['criterion'])
    rf_best['criterion'] = map_criterion_pso(rf_best['criterion'])

    dt_best, rf_best = float_hypers_to_int(dt_best, rf_best)

    best_config = [dt_best, rf_best]
    compute_time = [dt_ct, rf_ct]

    return best_config, compute_time

def bo_tpe(X_train, y_train, num_folds=fold, max_evals=max_evals, show_result=show_result):
    """ Optimize using Bayesian Optimization with Tree-parzen estimator (BO-TPE) via Hyperopt 

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The data to fit. Can be for example a list, or an array.
    y : array-like of shape (n_samples,)
        The target variable to try to predict
    num_folds : int 
        Number of folds of the KFold 
    max_evals : int 
        Number of max evaluation function of the optimization method
    show_process : bool
        The toggle for printing out the process
    
    Return 
    ------
    best_config : list 
        List of best hyperparameters configuration of DT and RF models 
    compute_time : list 
        List of of computational time of DT and RF in second 

    Example 
    -------
    >>> bo_best[smell], bo_ct[smell] = bo_tpe(X_train, y_train, max_evals=50)
    """

    print("Optmize using BO-TPE . . . ", end=" ")

    # ----------------- decision tree -----------------
    st = time.time()

    # objective function for decision_tree model
    def dt_objective(params):
        params = {
            'criterion': params['criterion'],
            'max_depth': int(params['max_depth']),
            'min_samples_split': int(params['min_samples_split']),
            'min_samples_leaf': int(params['min_samples_leaf']),
            'max_features': int(params['max_features'])
        }
        clf = DecisionTreeClassifier(**params)
        score = mean(cross_val_score(clf, X_train, y_train, scoring='recall', cv=StratifiedKFold(
            n_splits=num_folds), n_jobs=-1))
        # invert recall score to be negative value for minimize problem
        return -score
    
    # configuration space
    dt_space = {
        'criterion': hp.choice('criterion', ['gini', 'entropy']),
        'max_depth': hp.quniform('max_depth', 5, 50, 1),
        'min_samples_split': hp.quniform('min_samples_split', 2, 11, 1),
        'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 11, 1),
        'max_features': hp.quniform('max_features', 1, 64, 1)
    }

    # optimizing using fmin()
    dt_best = fmin(fn=dt_objective,
                    space=dt_space,
                    algo=tpe.suggest,
                    max_evals=max_evals,
                    return_argmin=False)

    et = time.time()
    dt_ct = et - st

    # ----------------- random forest -----------------
    st = time.time()

    # objective function for random_forest model
    def rf_objective(params):
        params = {
            'criterion': str(params['criterion']),
            'max_depth': int(params['max_depth']),
            'min_samples_split': int(params['min_samples_split']),
            'min_samples_leaf': int(params['min_samples_leaf']),
            'max_features': int(params['max_features']),
            'n_estimators': int(params['n_estimators'])
        }
        clf = RandomForestClassifier(**params)
        score = mean(cross_val_score(clf, X_train, y_train, scoring='recall', cv=StratifiedKFold(
            n_splits=num_folds), n_jobs=-1))

        return -score

    # configuration space
    rf_space = {
        'criterion': hp.choice('rf_criterion', ['gini', 'entropy']),
        'max_depth': hp.quniform('rf_max_depth', 5, 50, 1),
        'min_samples_split': hp.quniform('rf_min_samples_split', 2, 11, 1),
        'min_samples_leaf': hp.quniform('rf_min_samples_leaf', 1, 11, 1),
        'max_features': hp.quniform('rf_max_features', 1, 64, 1),
        'n_estimators': hp.quniform('rf_n_estimators', 100, 300, 5)
    }
    
    # optimizing using fmin()
    rf_best = fmin(fn=rf_objective,
                    space=rf_space,
                    algo=tpe.suggest,
                    max_evals=max_evals,
                    return_argmin=False)

    et = time.time()
    rf_ct = et - st

    dt_best, rf_best = float_hypers_to_int(dt_best, rf_best)
    

    print("Done !\n")

    def print_best():
        print("\n-----------------------------")
        print("Decision Tree: Hyperopt estimated optimum {}".format(dt_best))
        print("Random Forest: Hyperopt estimated optimum {}".format(rf_best))
        print("------------------------------\n")

    if show_result:
        print_best()
    
    best_config = [dt_best, rf_best]
    compute_time = [dt_ct, rf_ct]

    return best_config, compute_time

def export_hpo_compute_time(bo, pso, smac, files_path="../reports"):
    ''' Gets and exports the computational time of HPO methods

    Parameters
    ----------
    bo : dict 
        The computational time of BO-TPE for decision tree and random forest models stored in list -> [dt_ct, rf_ct]
    pso : dict
        The computational time of PSO for decision tree and random forest models stored in list -> [dt_ct, rf_ct]
    smac : dict
        The computational time of SMAC for decision tree and random forest models stored in list -> [dt_ct, rf_ct]
    files_path : str
        The file location to be exported
    '''

    bo_df = pd.DataFrame.from_dict(bo, orient='index', columns=["DT-BO", "RF-BO"])
    pso_df = pd.DataFrame.from_dict(pso, orient='index', columns=["DT-PSO", "RF-PSO"])
    smac_df = pd.DataFrame.from_dict(smac, orient='index', columns=["DT-SMAC", "RF-SMAC"])

    # concat all HPO into one dataframe and export to csv 
    df = pd.concat([bo_df, pso_df, smac_df], axis=1)
    df.to_csv(files_path)
