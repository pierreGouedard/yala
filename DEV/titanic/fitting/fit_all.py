# Global import
import os
import pandas as pd
import time
import numpy as np
import json

# Local import
from settings import features_path, export_path
from src.dev.prediction import ClassifierSelector

# Declare input and outputs
inputs = {'train': {'path': features_path, 'name': 'train.csv'}}
outputs = {'result': {'path': os.path.join(export_path, 'yala_simulation'), 'name': 'ts={}.json'}}
parameters = {}

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

# Parameters for decision tree
params_dt = {'criterion': 'entropy', 'min_samples_split': 2, 'max_depth': 35}

# Parameters for random forest
params_rf = {'n_jobs': 2, 'criterion': 'entropy', 'min_samples_split': 2, 'max_depth': 35, 'n_estimators': 200}

# Parameters for xgboost
params_xgb = {
    'nthread': 2, 'n_jobs': 2, 'objective': "binary:logistic", "verbosity": 0, 'lambda': 0,
    'min_child_leaf': 3, "colsample_bytree": 0.8, "subsample": 0.8, "learning_rate": 0.05, 'gamma': 2,
    'alpha': 1e-4, "max_depth": 3, "n_estimators": 200
}

# Parameters for Yala
params_yala = {
    'sampling_rate': 0.8, 'n_sampling': 10, 'max_iter': 20, 'learning_rate': 5e-2, 'batch_size': 800,
    'drainer_batch_size': 800, 'min_firing': 15, 'min_precision': 0.75, 'max_retry': 1
}
params_encoding = {
    'params_num_enc': {'n_bins': 20, 'method': 'signal'},
    'params_cat_enc': {'sparse': True, 'dtype': bool, 'handle_unknown': 'ignore'},
}

# Load data
df_train = pd.read_csv(os.path.join(inputs['train']['path'], inputs['train']['name']), index_col=None)

# set seed randomly and init output
seed = np.random.randint(0, 1000)
d_output = {"yala": {}, "dt": {}, "rf": {}, "xgb": {}}

# Fit YALA and get scores
print("YALA \n ------------- \n")
t0 = time.time()
np.random.seed(seed)
try:
    cs = ClassifierSelector(
        df_data=df_train,
        model_classification='yala',
        params_features=params_encoding,
        params_features_grid={},
        params_mdl=params_yala,
        params_mdl_grid={},
        params_fold=params_folds,
        scoring='accuracy'
    )
    _, _, d_scores = cs.fit().get_classifier().evaluate(cs.fold_manager.df_train, cs.fold_manager.df_test)
    print("Duration of algorithm is: {} \n Score for YALA is: {}".format(time.time() - t0, d_scores))
    d_output['yala'] = {"time": time.time() - t0, "score": d_scores}

except (ValueError, IndexError, KeyError, AssertionError) as e:
    print("Fail ! \n {} ".format(e))
    d_output['yala'] = {"time": time.time() - t0, "score": "fail"}
print("\n ------------- \n")

# Fit Benchmark
params_folds['params_builder'].update({'target_transform': 'encoding', 'method': 'cat_encode'})

# Fit DT and get scores
print("DT \n ------------- \n")
t0 = time.time()
np.random.seed(seed)
cs = ClassifierSelector(
    df_data=df_train,
    model_classification='dt',
    params_features={'sparse': False, 'dtype': int},
    params_features_grid={},
    params_mdl=params_dt,
    params_mdl_grid={},
    params_fold=params_folds,
    scoring='accuracy'
)
_, _, d_scores = cs.fit().get_classifier().evaluate(cs.fold_manager.df_train, cs.fold_manager.df_test)
print("Duration of algorithm is: {} \n Score is: {}".format(time.time() - t0, d_scores))
print("\n ------------- \n")
d_output['dt'] = {"time": time.time() - t0, "score": d_scores}


# Fit RF and get scores
print("RF \n ------------- \n")
t0 = time.time()
np.random.seed(seed)
cs = ClassifierSelector(
    df_data=df_train,
    model_classification='rf',
    params_features={'sparse': False, 'dtype': int},
    params_features_grid={},
    params_mdl=params_rf,
    params_mdl_grid={},
    params_fold=params_folds,
    scoring='accuracy'
)
_, _, d_scores = cs.fit().get_classifier().evaluate(cs.fold_manager.df_train, cs.fold_manager.df_test)
print("Duration of algorithm is: {} \n Score is: {}".format(time.time() - t0, d_scores))
print("\n ------------- \n")
d_output['rf'] = {"time": time.time() - t0, "score": d_scores}


# Fit XGBOOST and get scores
print("XGB \n ------------- \n")
t0 = time.time()
np.random.seed(seed)
cs = ClassifierSelector(
    df_data=df_train,
    model_classification='xgb',
    params_features={'sparse': False, 'dtype': int},
    params_features_grid={},
    params_mdl=params_xgb,
    params_mdl_grid={},
    params_fold=params_folds,
    scoring='accuracy'
)
_, _, d_scores = cs.fit().get_classifier().evaluate(cs.fold_manager.df_train, cs.fold_manager.df_test)
print("Duration of algorithm is: {} \n Score is: {}".format(time.time() - t0, d_scores))
print("\n ------------- \n")
d_output['xgb'] = {"time": time.time() - t0, "score": d_scores}

# Write result
ts = int(time.process_time()*1000)
with open(os.path.join(outputs['result']['path'], outputs['result']['name'].format(ts)), 'w') as handle:
    json.dump(str(d_output), handle)
