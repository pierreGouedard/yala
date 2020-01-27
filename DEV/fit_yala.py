# Global import
import os
import pandas as pd

# Local import
from settings import models_path, features_path
from src.utils.names import KVName
from src.utils.prediction import ClassifierSelector

# Declare input and outputs
inputs = {
    'train': {'path': features_path, 'name': 'train.csv'}
}
outputs = {
    'model': {'path': models_path, 'name': 'yala.pickle'},
}

parameters = {
    'model': 'dt'
}

# Set params for cross validation
params_folds = {
    'nb_folds': 4, 'method': 'standard',
    'params_builder': {
        'method': 'cat_num_encode',
        'cat_cols': ['sex', 'cabin_letter', 'embarked', 'ticket_letter'],
        'num_cols': ['pclass', 'age', 'sibsp', 'parch', 'fare', 'cabin_num', 'ticket_num'],
        'target_tranform': 'sparse_boolean'
    }
}

# Parameters for decision tree
sampling_rate = 0.8,
n_sampled_vertices = 10,
max_iter = 10,
learning_rate = 5e-2,
p_flip = 0.5,
batch_size = 500,
firing_graph = None,
t = None,
min_firing = 10

params_yala = {
    'sampling_rate': 0.8, 'n_sampled_vertices': 2, 'max_iter': 2, 'learning_rate': 2e-1, 'p_flip': 0.,
    'batch_size': 800, 'min_firing': 15
}
params_yala_grid = {}
params_encoding = {
    'params_num_enc': {'n_bins': 20, 'method': 'signal'},
    'params_cat_enc': {'sparse': True, 'dtype': bool, 'handle_unknown': 'ignore'},
}

# Load data
df_train = pd.read_csv(os.path.join(inputs['train']['path'], inputs['train']['name']), index_col=None)

cs = ClassifierSelector(
    df_data=df_train,
    model_classification='yala',
    params_features=params_encoding,
    params_features_grid={},
    params_mdl=params_yala,
    params_mdl_grid=params_yala_grid,
    params_fold=params_folds,
    scoring='accuracy'
)
# Fit and save modeltrain
cs.fit().save_classifier(os.path.join(outputs['model']['path'], outputs['model']['name']))




