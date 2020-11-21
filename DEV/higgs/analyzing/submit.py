# Global import
import os
import pandas as pd
import pickle
import numpy as np
import sys
sys.path.append(os.getcwd())

# Local import
from settings import models_path, features_path, submission_path
from src.dev.names import KVName
from src.tools.ams_metric import AMS_metric

# Declare input and outputs
inputs = {
    'model': {'path': models_path, 'name': 'higgs/{}.pickle'},
    'test': {'path': features_path, 'name': 'higgs/test.csv'},
}
outputs = {
    'submission': {'path': submission_path, 'name': 'higgs/submission_{}.csv'},

}

parameters = {
    'model': 'yala',
    'id': 9,
    'n_label': 1

}

name_mdl = '{}'.format(KVName.from_dict({'model': parameters['model']}).to_string())

# Load model
with open(os.path.join(inputs['model']['path'], inputs['model']['name']).format(name_mdl), 'rb') as handle:
    classifier = pickle.load(handle)

# Load test features
df_test = pd.read_csv(os.path.join(inputs['test']['path'], inputs['test']['name']), index_col="EventId")
import IPython
IPython.embed()
import time
t0 = time.time()
df_probas = classifier.predict_proba(df_test, **{'n_label': parameters['n_label']})

print('duration predict {}'.format(time.time() - t0))

if parameters['n_label'] == 1:
    df_probas = df_probas.loc[:, 's']\
            .sort_values(ascending=False)\
            .to_frame('Class')\
            .assign(rank=list(range(len(df_test))))\
            .assign(
                RankOrder=range(len(df_test), 0, -1)
            ) \
            .reset_index() \
            .loc[:, ['EventId', 'RankOrder', 'Class']]

else:
    df_probas = df_probas.loc[:, 's']\
        .sort_values(ascending=False) \
        .to_frame('Class') \
        .assign(rank=list(range(len(df_test)))) \
        .assign(
            RankOrder=range(len(df_test), 0, -1)
        ) \
        .reset_index() \
        .loc[:, ['EventId', 'RankOrder', 'Class']]


import IPython
IPython.embed()
for t in [0.14, 0.15, 0.16]:
    parameters['treshold'] = t
    name_submission = '{}'.format(KVName.from_dict(parameters).to_string())
    submission_path = os.path.join(outputs['submission']['path'], outputs['submission']['name']).format(name_submission)

    df_probas.assign(
        Class=lambda df: np.where(df['Class'] > df['Class'].iloc[int(len(df) * parameters['treshold'])], 's', 'b')
    ).to_csv(submission_path, index=None)

