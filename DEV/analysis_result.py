# Global import
import os
import pandas as pd
import json
import ast
import numpy as np

# Local import
from settings import export_path

# Declare input and outputs
inputs = {
    'new': {'path': export_path, 'name': 'yala_simulation/new'},
    'old': {'path': export_path, 'name': 'yala_simulation/old'}
}
outputs = {}

parameters = {
    'model_keys': ["yala", "dt", "rf", "xgb"]
}


def load_result(path):
    l_results, n_fail = [], 0

    for file_name in os.listdir(path):
        with open(os.path.join(path, file_name), 'rb') as handle:
            d_scores = ast.literal_eval(json.load(handle))

        for model_name, d_score in d_scores.items():
            if d_score['score'] == 'fail':
                n_fail += 1
                continue

            l_results.append(
                {'model': model_name, 'train_accuracy': d_score['score']['train']['accuracy'],
                 'test_accuracy': d_score['score']['test']['accuracy'], 'time': d_score['time']}
            )

    return pd.DataFrame.from_records(l_results), n_fail


# Load result
df_result_new, n_fail_new = load_result(os.path.join(inputs['new']['path'], inputs['new']['name']))
df_result_new = df_result_new.astype({"train_accuracy": float, "test_accuracy": float, "time": float})

df_result_old, n_fail_old = load_result(os.path.join(inputs['old']['path'], inputs['old']['name']))
df_result_old = df_result_old.astype({"train_accuracy": float, "test_accuracy": float, "time": float})

# stats
df_result_new.groupby('model').agg([np.mean, min, max, np.std])
df_result_old.groupby('model').agg([np.mean, min, max, np.std])

import IPython
IPython.embed()
