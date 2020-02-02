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
    'model': {'path': models_path, 'name': 'model=yala.pickle'},
}

parameters = {
    'model': 'dt'
}

# Set params for cross validation
params_folds = {
    'nb_folds': 3,
    'method': 'standard',
    'params_builder': {
        'method': 'cat_num_encode',
        'cat_cols': ['sex', 'cabin_letter', 'embarked', 'ticket_letter'],
        'num_cols': ['pclass', 'age', 'sibsp', 'parch', 'fare', 'cabin_num', 'ticket_num'],
        'target_tranform': 'sparse_boolean'
    }
}

params_yala = {
    'sampling_rate': 0.8, 'n_sampled_vertices': 5, 'max_iter': 2, 'learning_rate': 2e-1, 'batch_size': 800,
    'min_firing': 15
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

# Select, fit and get trained classifier
classifier = cs.fit().get_classifier()

# Evaluate on our own test
confmat_train, confmat_test, d_scores = classifier.evaluate(cs.fold_manager.df_train, cs.fold_manager.df_test)

# Save classifier
cs.save_classifier(os.path.join(outputs['model']['path'], outputs['model']['name']))
