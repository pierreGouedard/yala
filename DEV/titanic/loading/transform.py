# Global import
import os
import pandas as pd

# Local import
from settings import raw_path, features_path
from src.dev.utils import prepare_titanic_data

# Declare input and outputs
inputs = {
    'train': {'path': raw_path, 'name': 'train.csv'},
    'test': {'path': raw_path, 'name': 'test.csv'}
}
outputs = {
    'train': {'path': features_path, 'name': 'train.csv'},
    'test': {'path': features_path, 'name': 'test.csv'}
}
parameters = {
    'features': ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked', 'Ticket'],
    'target': ['Survived'],
    'cols_out': [
        'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'cabin_letter', 'cabin_num', 'embarked',
        'ticket_letter', 'ticket_num'
    ]
}

# Load and save dataset
l_cols = parameters['features'] + parameters['target']

df_train = pd.read_csv(
    os.path.join(inputs['train']['path'], inputs['train']['name']), header=0,
    usecols=parameters['features'] + parameters['target']
)
df_test = pd.read_csv(os.path.join(inputs['test']['path'], inputs['test']['name']), header=0,
                      usecols=parameters['features'])

# Prepare data
df_train = prepare_titanic_data(df_train)
df_test = prepare_titanic_data(df_test)

df_train.rename(columns={parameters['target'][0].lower(): 'target'}, inplace=True)
df_train = df_train[parameters['cols_out'] + ['target']]
df_test = df_test[['passengerid'] + parameters['cols_out']]

df_train.to_csv(os.path.join(outputs['train']['path'], outputs['train']['name']), index=None)
df_test.to_csv(os.path.join(outputs['test']['path'], outputs['test']['name']), index=None)



