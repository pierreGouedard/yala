import kaggle
import os
import pandas as pd
import pickle as pckl
import numpy as np

from settings import models_path, submission_path, features_path
inputs = {
    'model': {'path': models_path, 'name': ''},
    'test': {'path': features_path, 'name': 'higgs/test.csv'},

}
outputs = {
    'submission': {'path': submission_path, 'name': '{}.csv'},

}

l_names = [
    "higgs/search/model_min_firing=300.0,min_precision=0.8.pickle",
    "higgs/search/model_min_firing=300.0,min_precision=0.95.pickle",
    "higgs/search/model_min_precision=0.7.pickle",
    "higgs/search/model_min_precision=0.9.pickle",
    "higgs/search/model_min_precision=0.75.pickle",
    "higgs/search/model_min_precision=0.85.pickle"
]
test_path = os.path.join(inputs['test']['path'], inputs['test']['name'])
df_test = pd.read_csv(test_path, index_col="EventId")

for name in l_names:

    # Set paths
    models_path = os.path.join(inputs['model']['path'], name)
    name_submission = outputs['submission']['name'].format(name.replace('.pickle', ''))
    submission_path = os.path.join(outputs['submission']['path'], name_submission)

    # Load model
    with open(models_path, 'rb') as handle:
        classifier = pckl.load(handle)

    #if os.path.exists(submission_path):
    #    continue

    # Predict and save to csv file
    df_probas = classifier.predict_proba(df_test, **{'n_label': 2, 'max_batch': 30000})\
        .loc[:, 's']\
        .sort_values(ascending=False) \
        .to_frame('Class') \
        .assign(rank=list(range(len(df_test)))) \
        .assign(
            RankOrder=range(len(df_test), 0, -1)
        ) \
        .reset_index() \
        .loc[:, ['EventId', 'RankOrder', 'Class']]\
        .assign(
            Class=lambda df: np.where(df['Class'] > df['Class'].iloc[int(len(df) * 0.15)], 's', 'b')
        ).to_csv(submission_path, index=None)

