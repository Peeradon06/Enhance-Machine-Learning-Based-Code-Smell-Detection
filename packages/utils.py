import os
from os import path
import sys
sys.path.append('../')
from config.definitions import FONTANA_DIR
import pandas as pd
import numpy as np

def export_report(models_dict, dir="/reports/baseline.csv"):
    """ Export models result reports
    
    Parameters 
    ----------
    models_dict : dict
        Dictionary contains model's dictionary, for example, 
        baseline_model = {"GOD_CLASS", [DT_classifier, RF_classifier]}
    dir : string 
        Path of the exported reports (always export in .csv files)
    
    Example
    -------
    >>> export_report(baseline_models, dir='/qc-baseline-result/baseline.csv')
    """

    # validation metrics
    acc = []
    pre = []
    rec = []
    f1 = []
    idx = []
    roc = []
    training_time = []
    
    # prediciton metrics 
    pred_acc = []
    pred_pre = []
    pred_rec = []
    pred_f1 = []
    pred_roc = []

    # loop through each smell dataset
    for k,v in models_dict.items():
        # each model (DT and RF)
        for model in models_dict[k]:
            # store the name as an index
            idx.append(model.model_name)      

            # validataion set
            acc.append(model.accuracy)
            pre.append(model.precision)
            rec.append(model.recall)
            f1.append(model.f1)
            roc.append(model.roc_auc)
            training_time.append(model.training_time)

            # test set
            pred_acc.append(model.pred_accuracy)
            pred_pre.append(model.pred_precision)
            pred_rec.append(model.pred_recall)
            pred_f1.append(model.pred_f1)
            pred_roc.append(model.pred_roc)
    
    data = {
    'Accuracy' : acc,
    'Precision' : pre,
    'Recall' : rec,
    'F1' : f1,
    'ROC' : roc,
    'Test_Accuracy' : pred_acc,
    'Test_Precision' : pred_pre,
    'Test_Recall' : pred_rec,
    'Test_F1' : pred_f1,
    'Test_ROC' : pred_roc,
    'Training_time' : training_time,
    }
    
    df = pd.DataFrame(data, index=idx)
    
    # check for exist directory 
    dirs = ['', 'qc', 'mlcq', 'cross']
    for d in dirs:
        dir_path = '../reports/{}'.format(d)
        if path.isdir(dir_path) == False:
            os.makedirs(dir_path)

    try:
        df.to_csv('../reports'+dir)
    except Exception as e:
        print('Error while exporting the report ', e)
    
    print('Report exported ! at /reports{}'.format(dir))


def read_QC_train_test(data_type="train", class_col=None, method_col=None):
    """ Read QC train and test datasets 

    Parameters
    ----------
    data_type : str (lower or upper case)
        String for selecting dataset between ['train', 'test']
    class_col : list of string
        List contains column name to be slected from god_class and data_class dataset
    method_col : list of string
        List contains column name to be selected from feature_envy and long_method dataset
    
    Return
    ------
    datasets : dict 
        Dictonary contains four code smells which keys are the name and values are dataframe
    
    Example
    -------
    >>> train_data = read_QC_train_test("train")
    """

    data_type = data_type.lower()
    
    try:
        data_class = pd.read_csv(FONTANA_DIR+"data_class_{}.csv".format(data_type))
        feature_envy = pd.read_csv(FONTANA_DIR+"feature_envy_{}.csv".format(data_type))
        god_class = pd.read_csv(FONTANA_DIR+"god_class_{}.csv".format(data_type))
        long_method = pd.read_csv(FONTANA_DIR+"long_method_{}.csv".format(data_type))

        # read specific columns
        if class_col:
            data_class = pd.read_csv(FONTANA_DIR+"data_class_{}.csv".format(data_type),usecols=class_col)
            god_class = pd.read_csv(FONTANA_DIR+"god_class_{}.csv".format(data_type), usecols=class_col)
        if method_col:
            feature_envy = pd.read_csv(FONTANA_DIR+"feature_envy_{}.csv".format(data_type))
            long_method = pd.read_csv(FONTANA_DIR+"long_method_{}.csv".format(data_type))

    except FileNotFoundError as e:
        print("FileNotFoundError while reading QC training dataset")
    
    datasets = {
        "DATA_CLASS": data_class,
        "FEATURE_ENVY": feature_envy,
        "GOD_CLASS": god_class,
        "LONG_METHOD": long_method
    }

    print("All QC {}ing datasets are loaded !".format(data_type))
    
    return datasets


def read_QC(class_col=None, method_col=None):
    """ Read QC dataset 
    
    Parameters
    ----------
    class_col : list of string
        List contains column name to be slected from god_class and data_class dataset
    method_col : list of string
        List contains column name to be selected from feature_envy and long_method dataset
    
    Return
    ------
    datasets : dict 
        Dictonary contains four code smells which keys are the name and values are dataframe
    
    Example 
    -------
    >>> QC_dataset = read_QC()
    >>> QC_dataset = read_QC(class_col= ["DIT_type", "LOC_type"], method_col=["LOC_method"])
    """
    
    # clean the data
    def clean_qc(dataset):

        for smell in dataset:
            # drop null instance
            dataset[smell] = dataset[smell].replace('?', np.nan).dropna()

        return dataset

    data_class = pd.read_csv(FONTANA_DIR+"data_class.csv")
    feature_envy = pd.read_csv(FONTANA_DIR+"feature_envy.csv")
    god_class = pd.read_csv(FONTANA_DIR+"god_class.csv")
    long_met = pd.read_csv(FONTANA_DIR+"long_method.csv")
    
    # read specific columns
    if class_col:
        data_class = pd.read_csv(FONTANA_DIR+"data_class.csv", usecols=class_col)
        god_class = pd.read_csv(FONTANA_DIR+"god_class.csv", usecols=class_col)

    if method_col:
        feature_envy = pd.read_csv(FONTANA_DIR+"feature_envy.csv", usecols=method_col)
        long_met = pd.read_csv(FONTANA_DIR+"long_method.csv", usecols=method_col)

    datasets = {
        "DATA_CLASS": data_class,
        "FEATURE_ENVY": feature_envy,
        "GOD_CLASS": god_class,
        "LONG_METHOD": long_met
    }

    clean_datasets = clean_qc(datasets)

    print("All QC datasets are loaded !")

    return clean_datasets