import os

yala_home = os.path.join(os.path.expanduser("~"), 'yala')
data_path = os.path.join(yala_home, 'DATA')

raw_path = os.path.join(data_path, 'raw')
features_path = os.path.join(data_path, 'features')
models_path = os.path.join(data_path, 'models')
submission_path = os.path.join(data_path, 'submissions')
