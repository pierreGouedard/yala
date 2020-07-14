# Global import
import os
import pandas as pd
import argparse
import re
import json
import numpy as np

# Local import
from settings import models_path, features_path
from src.dev.names import KVName
from src.tools.ams_metric import embedded_ams
from src.dev.prediction import ClassifierSelector

# Declare input and outputs
inputs = {
    'train': {'path': features_path, 'name': 'higgs/train.csv'},
    'weights': {'path': features_path, 'name': 'higgs/weights.csv'}
}
outputs = {
    'model': {'path': models_path, 'name': 'higgs/search/model_{}.pickle'},
    'stats': {'path': models_path, 'name': 'higgs/search/stats_{}.json'},
}

parameters = {
    'model': 'yala'
}

# Set params for cross validation
params_folds = {
    'nb_folds': 2,
    'method': 'standard',
    "scoring": embedded_ams,
    'test_size': 0.01,
    'params_builder': {
        'method': 'cat_num_encode',
        'num_cols': [
            'DER_mass_MMC', 'DER_mass_transverse_met_lep', 'DER_mass_vis', 'DER_pt_h', 'DER_deltaeta_jet_jet',
            'DER_mass_jet_jet', 'DER_prodeta_jet_jet', 'DER_deltar_tau_lep', 'DER_pt_tot', 'DER_sum_pt',
            'DER_pt_ratio_lep_tau', 'DER_met_phi_centrality', 'PRI_tau_pt', 'PRI_tau_pt', 'PRI_tau_eta', 'PRI_tau_phi',
            'PRI_lep_pt', 'PRI_lep_eta', 'PRI_lep_phi', 'PRI_met', 'PRI_met_phi', 'PRI_met_sumet', 'PRI_jet_leading_pt',
            'PRI_jet_leading_eta', 'PRI_jet_subleading_eta', 'PRI_jet_all_pt'
        ],
        'cat_cols': ['PRI_jet_num'],
        'target_transform': 'sparse_encoding',
        'n_label': 2
    }
}

# Get params from args
params_yala = {
    'sampling_rate': 1.0, 'max_iter': 100, 'min_gain': 1e-3, 'batch_size': 90000,
    'drainer_batch_size': 30000, 'min_firing': 1000, 'min_precision': 0.7, 'max_retry': 5,
    'dropout_mask': 0.5, "max_candidate": 100
}
params_yala_grid = {}

params_encoding = {
    'params_num_enc': {'n_bins': 30, 'method': 'signal'},
    'params_cat_enc': {'sparse': True, 'dtype': bool, 'handle_unknown': 'ignore'},
}
np.random.seed(1234)


def fit_model(model_path, stats_path):
    import time
    t0 = time.time()

    # Load inputs
    df_train = pd.read_csv(os.path.join(inputs['train']['path'], inputs['train']['name']), index_col='EventId')
    df_weights = pd.read_csv(os.path.join(inputs['weights']['path'], inputs['weights']['name']), index_col="EventId")

    cs = ClassifierSelector(
        df_data=df_train,
        df_weights=df_weights,
        model_classification='yala',
        params_features=params_encoding,
        params_features_grid={},
        params_mdl=params_yala,
        params_mdl_grid=params_yala_grid,
        params_fold=params_folds,
        scoring='accuracy'
    )

    # Select, fit and get and save trained classifier
    classifier = cs.fit().save_classifier(model_path).get_classifier()
    d_stats = {'execution_time': time.time() - t0}

    l_partitions = classifier.model_classification.firing_graph.partitions
    for i in range(2):
        l_partitions_sub = [p for p in l_partitions if p['label_id'] == i]
        d_stats.update({
            "nb_vertex_{}".format(i): len(l_partitions_sub),
            "mean_precision_{}".format(i): np.mean([p['precision'] for p in l_partitions_sub]),
            "min_precision_{}".format(i): np.min([p['precision'] for p in l_partitions_sub]),
            "max_precision_{}".format(i): np.max([p['precision'] for p in l_partitions_sub])
        })

    with open(stats_path, 'w') as handle:
        json.dump(d_stats, handle)


def parse_args():
    pattern_args = re.compile(r"^[aA-zZ]*=([aA-zZ]|[0-9]|\.)*$")
    parser = argparse.ArgumentParser(description="search")

    # Add parameter parser
    parser.add_argument(
        "-p",
        "--parameters",
        dest="params",
        nargs="*",
        default=[],
        help="Pass the space separated string <param_key>=<param_value> default is []",
    )
    args = parser.parse_args()

    # Make sure parameters arg are correct
    for param in args.params:
        if not pattern_args.match(param):
            raise ValueError("parameters should be passed as space separated string <param_key>=<param_value>")

    return {param.split('=')[0]: float(param.split('=')[1]) for param in args.params}


if __name__ == "__main__":

    params_yala_ = parse_args()
    params_yala.update(params_yala_)

    # Defined custom output name based on parameters of the script
    name_mdl = '{}'.format(KVName.from_dict(params_yala_).to_string())
    path_mdl = os.path.join(outputs['model']['path'], outputs['model']['name'].format(name_mdl))
    path_stats = os.path.join(outputs['stats']['path'], outputs['stats']['name'].format(name_mdl))

    if not os.path.exists(path_mdl):
        fit_model(path_mdl, path_stats)
