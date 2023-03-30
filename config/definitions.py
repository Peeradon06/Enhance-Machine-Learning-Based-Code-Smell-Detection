import os

# This files contains paths variables

# root 
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
# fontana dataset 
FONTANA_DIR = os.path.join(ROOT_DIR, 'data/fontana-QC/')
# models
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
# models' hyper-parameter configs from HPOs 
MODEL_CONF_DIR = os.path.join(ROOT_DIR, 'models_conf')
# MLCQ dataset
MLCQ_DIR = os.path.join(ROOT_DIR, 'data/mlcq/')
# mapped dataset of MLCQ and QC dir
MAPPED_DATASET_DIR = os.path.join(ROOT_DIR, 'data/mapped-datasets/')