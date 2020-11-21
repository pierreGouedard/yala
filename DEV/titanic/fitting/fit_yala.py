# Global import
import os
import pandas as pd
import json

# Local import
from settings import models_path, features_path, export_path
from src.dev.names import KVName
from src.dev.prediction import ClassifierSelector

# Declare input and outputs
inputs = {
    'train': {'path': features_path, 'name': 'titanic/train.csv'}
}
outputs = {
    'model': {'path': models_path, 'name': 'greedy.pickle'},
}

parameters = {
    'model': 'yala2'
}

# Set params for cross validation
params_folds = {
    'nb_folds': 2,
    'method': 'standard',
    'params_builder': {
        'method': 'cat_num_encode',
        'cat_cols': ['sex', 'cabin_letter', 'embarked', 'ticket_letter'],
        'num_cols': ['pclass', 'age', 'sibsp', 'parch', 'fare', 'cabin_num', 'ticket_num'],
        'target_transform': 'sparse_encoding'
    }
}

params_yala = {
    'sampling_rate': 0.8, 'n_sampling': 10, 'max_iter': 20, 'learning_rate': 5e-2, 'batch_size': 800,
    'drainer_batch_size': 800, 'min_firing': 15, 'min_precision': 0.75, 'max_retry': 1
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
print(d_scores)
name_export = '{}'.format(KVName.from_dict(parameters).to_string())

# Export metrics and predicion
with open(os.path.join(export_path, 'scores_{}.json'.format(name_export)), 'w') as handle:
    json.dump(str(d_scores), handle)

cs.fold_manager.df_train\
    .join(classifier.predict(cs.fold_manager.df_train))\
    .to_csv(os.path.join(export_path, 'preds_train_{}.csv'.format(name_export)), index=None)

cs.fold_manager.df_test\
    .join(classifier.predict(cs.fold_manager.df_test))\
    .to_csv(os.path.join(export_path, 'preds_test_{}.csv'.format(name_export)), index=None)

# Save classifier
cs.save_classifier(os.path.join(outputs['model']['path'], outputs['model']['name']))

