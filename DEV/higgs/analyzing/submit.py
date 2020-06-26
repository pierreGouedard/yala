# Global import
import os
import pandas as pd
import pickle
import numpy as np
import time
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
    'treshold': 0.8,
    'id': 13#int(time.process_time()*1000)

}

name_mdl = '{}'.format(KVName.from_dict({'model': parameters['model']}).to_string())

# Load model
with open(os.path.join(inputs['model']['path'], inputs['model']['name']).format(name_mdl), 'rb') as handle:
    classifier = pickle.load(handle)

# Load test features
df_test = pd.read_csv(os.path.join(inputs['test']['path'], inputs['test']['name']), index_col="EventId")

import time
t0 = time.time()
import IPython
IPython.embed()
df_probas = classifier.predict_proba(df_test)

print('duration predict {}'.format(time.time() - t0))

df_probas = df_probas.loc[:, 's']\
        .sort_values(ascending=False)\
        .to_frame('Class')\
        .assign(rank=list(range(len(df_test))))\
        .assign(
            RankOrder=range(len(df_test), 0, -1)
        ) \
        .reset_index() \
        .loc[:, ['EventId', 'RankOrder', 'Class']]

for t in [0.12, 0.13, 0.14, 0.15, 0.16, 0.17]:
    parameters['treshold'] = t
    name_submission = '{}'.format(KVName.from_dict(parameters).to_string())
    submission_path = os.path.join(outputs['submission']['path'], outputs['submission']['name']).format(name_submission)

    df_probas.assign(
        Class=lambda df: np.where(df['Class'] > df['Class'].iloc[int(len(df) * parameters['treshold'])], 's', 'b')
    ).to_csv(submission_path, index=None)

