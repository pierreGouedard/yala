# Global import
import os
import pandas as pd

# Local import
from settings import models_path, features_path, submission_path
from src.dev.names import KVName
from src.dev.prediction import Classifier

# Declare input and outputs
inputs = {
    'test': {'path': features_path, 'name': 'test.csv'},
    'model': {'path': models_path, 'name': ''},
}
outputs = {
    'submission': {'path': submission_path, 'name': ''},
}

parameters = {
    'model': 'yala'
}

name_submission = '{}.csv'.format(KVName.from_dict(parameters).to_string())
name_mdl = '{}.pickle'.format(KVName.from_dict(parameters).to_string())

# Load test data
df_test = pd.read_csv(os.path.join(inputs['test']['path'], inputs['test']['name']), index_col='passengerid')

# Load model and predict test data
dc = Classifier.from_path(os.path.join(inputs['model']['path'], name_mdl))
s_preds = dc.predict(df_test)

# Save submission
df_submission = df_test.merge(s_preds, left_index=True, right_index=True)\
    .reset_index(drop=False)\
    .loc[:, ['passengerid', 'prediction']]\
    .rename(columns={'passengerid': 'PassengerId', 'prediction': 'Survived'})

df_submission.to_csv(os.path.join(outputs['submission']['path'], name_submission), index=None)
