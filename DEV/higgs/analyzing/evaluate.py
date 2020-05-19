# Global import
import os
import pandas as pd

# Local import
from settings import export_path
from src.dev.names import KVName
from src.tools.ams_metric import AMS_metric

# Declare input and outputs
inputs = {
    'submission': {'path': export_path, 'name': 'higgs/submission_{}.csv'},
    'solution': {'path': export_path, 'name': 'higgs/solution_{}.csv'},
}
outputs = {
}

parameters = {
    'model': 'yala'
}

path_submission = os.path.join(inputs['submission']['path'], inputs['submission']['name'])\
    .format(KVName.from_dict(parameters).to_string())
path_solution = os.path.join(inputs['solution']['path'], inputs['solution']['name'])\
    .format(KVName.from_dict(parameters).to_string())

AMS_metric(path_solution, path_submission)
