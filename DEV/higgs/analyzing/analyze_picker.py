# Global import
import os
import pandas as pd
import pickle
import sys
sys.path.append(os.getcwd())

# Local import
from settings import export_path, features_path


# Declare input and outputs
inputs = {
    'model': {'path': export_path, 'name': 'higgs/comparison_picker/{}.pickle'},
    'train': {'path': features_path, 'name': 'higgs/train.csv'},
}
outputs = {
}

# Load train data
df_train = pd.read_csv(os.path.join(inputs['train']['path'], inputs['train']['name']), index_col='EventId')

# Load classifier
# Load model
with open(os.path.join(inputs['model']['path'], inputs['model']['name'].format('orthogonal')), 'rb') as handle:
    orthogonal_classifier = pickle.load(handle)
with open(os.path.join(inputs['model']['path'], inputs['model']['name'].format('greedy')), 'rb') as handle:
    greedy_classifier = pickle.load(handle)


# Get train data and firing_graph
X, y = orthogonal_classifier.feature_builder.transform(df_train, target=True)
sax_map_fi = orthogonal_classifier.feature_builder.args['mapping_feature_input']

# Get firing graphs
greedy_fg = greedy_classifier.model_classification.firing_graph
orthogonal_fg = orthogonal_classifier.model_classification.firing_graph

yhat_greedy = greedy_fg.propagate(X)
yhat_orthogonal = orthogonal_fg.propagate(X)

inner = yhat_greedy.astype(int).T.dot(yhat_orthogonal)
norm_greedy, norm_orthogonal = yhat_greedy.sum(axis=0), yhat_orthogonal.sum(axis=0)

for i in range(2):
    print(f'overlap for label {i}')
    print(f'{inner[i, i]} / {norm_greedy[0, i]} overlapping fro greedy')
    print(f'{inner[i, i]} / {norm_orthogonal[0, i]} overlapping fro orthogonal')

for i in range(2):

    sax_i_ortho = orthogonal_fg.I[:, [p['indices'][0] for p in orthogonal_fg.partitions if p['label_id'] == i]]
    sax_i_greed = greedy_fg.I[:, [p['indices'][0] for p in greedy_fg.partitions if p['label_id'] == i]]

    ax_f_ortho = sax_map_fi.T.dot(sax_i_ortho).A
    ax_f_greedy = sax_map_fi.T.dot(sax_i_greed).A

    import IPython
    IPython.embed()