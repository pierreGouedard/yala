# Global import
import os
import pandas as pd
import json

# Local import
from settings import models_path, features_path, export_path
from src.utils.names import KVName
from src.utils.prediction import ClassifierSelector

# Declare input and outputs
inputs = {
    'train': {'path': features_path, 'name': 'train.csv'}
}
outputs = {
    'model': {'path': models_path, 'name': '{}.pickle'},
}

parameters = {
    'model': 'xgb'
}

# Set params for cross validation
params_folds = {
    'nb_folds': 2,
    'method': 'standard',
    'params_builder': {
        'method': 'cat_encode',
        'cat_cols': ['sex', 'cabin_letter', 'embarked', 'ticket_letter'],
        'target_transform': 'encoding'
    }
}

# Parameters for decision tree
params_dt = {'criterion': 'entropy', 'min_samples_split': 2, 'max_depth': 35}
params_dt_grid = {}

# Parameters for random forest
params_rf = {'n_jobs': 2, 'criterion': 'entropy', 'min_samples_split': 2, 'max_depth': 35, 'n_estimators': 200}
params_rf_grid = {}

# Parameters for xgboost
params_xgb = {
    'nthread': 2, 'n_jobs': 2, 'objective': "binary:logistic", "verbosity": 0, 'lambda': 0,
    'min_child_leaf': 3, "colsample_bytree": 0.8, "subsample": 0.8, "learning_rate": 0.05, 'gamma': 2,
    'alpha': 1e-4, "max_depth": 3, "n_estimators": 200

}
params_xgb_grid = {}

# Defined custom output name based on parameters of the script
name_mdl = '{}.pickle'.format(KVName.from_dict(parameters).to_string())
df_train = pd.read_csv(os.path.join(inputs['train']['path'], inputs['train']['name']), index_col=None)

# Instantiate document classifier
if parameters['model'] == 'xgb':
    cs = ClassifierSelector(
        df_data=df_train,
        model_classification=parameters['model'],
        params_features={'sparse': False, 'dtype': int},
        params_features_grid={},
        params_mdl=params_xgb,
        params_mdl_grid=params_xgb_grid,
        params_fold=params_folds,
        scoring='accuracy'
    )

elif parameters['model'] == 'dt':
    cs = ClassifierSelector(
        df_data=df_train,
        model_classification=parameters['model'],
        params_features={'sparse': False, 'dtype': int},
        params_features_grid={},
        params_mdl=params_dt,
        params_mdl_grid=params_dt_grid,
        params_fold=params_folds,
        scoring='accuracy'
    )

elif parameters['model'] == 'rf':
    cs = ClassifierSelector(
        df_data=df_train,
        model_classification=parameters['model'],
        params_features={'sparse': False, 'dtype': int},
        params_features_grid={},
        params_mdl=params_rf,
        params_mdl_grid=params_rf_grid,
        params_fold=params_folds,
        scoring='accuracy'
    )

else:
    raise ValueError('No valid model name set')

# Select, fit and get trained classifier
classifier = cs.fit().get_classifier()

# Evaluate on our own test
confmat_train, confmat_test, d_scores = classifier.evaluate(cs.fold_manager.df_train, cs.fold_manager.df_test)

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
cs.save_classifier(os.path.join(outputs['model']['path'], name_mdl))



