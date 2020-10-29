# Global import
import os
import pandas as pd
import pickle
import sys
sys.path.append(os.getcwd())

# Local import
from settings import models_path, features_path


# Declare input and outputs
inputs = {
    'model': {'path': models_path, 'name': 'higgs/model=yala.pickle'},
    'train': {'path': features_path, 'name': 'higgs/train.csv'},
}
outputs = {

}

# Load train data
df_train = pd.read_csv(os.path.join(inputs['train']['path'], inputs['train']['name']), index_col='EventId')

# Load classifier
# Load model
with open(os.path.join(inputs['model']['path'], inputs['model']['name']), 'rb') as handle:
    classifier = pickle.load(handle)

# Get train data and firing_graph
X, y = classifier.feature_builder.transform(df_train, target=True)
firing_graph = classifier.model_classification.firing_graph

# Isolate output of each vertex
l_outputs = [p['indices'][0] for p in firing_graph.partitions]
firing_graph = firing_graph.reset_output(l_outputs=l_outputs)

# Get prediction
yhat = firing_graph.propagate(X)

# Stat 1: Check that precision label and actual precision match
for p in firing_graph.partitions:
    precision = yhat.A[:, p['indices'][0]].astype(int).dot(y.A[:, p['label_id']]) / yhat.A[:, p['indices'][0]].sum()
    print(f"target precision is {p['precision']}, actual precision is {precision}")

# Stat 2: check similarity between vertices
ax_sum = yhat.A.sum(axis=0, keepdims=True).repeat(yhat.shape[1], axis=0)
ax_sum += yhat.A.sum(axis=0, keepdims=True).transpose()
ax_inner = yhat.transpose().astype(int).dot(yhat).A
ax_sim_old = ax_inner / (ax_sum - ax_inner)

print(f'Overall Coverage {(yhat.sum(axis=1).A > 0).sum()} out of {y.shape[0]}')
print(f'Mean Coverage {yhat.sum(axis=0).A.mean()} out of {y.shape[0]}')
print('Average number very similar vertices old {}'.format((ax_sim_old > 0.9).sum(axis=0).mean()))



