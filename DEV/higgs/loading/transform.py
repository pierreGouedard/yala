# Global import
import os
import pandas as pd

# Local import
from settings import raw_path, features_path
from src.dev.utils import prepare_higgs_data

# Declare input and outputs
inputs = {
    'train': {'path': raw_path, 'name': 'higgs/train.csv'},
    'test': {'path': raw_path, 'name': 'higgs/test.csv'}
}
outputs = {
    'train': {'path': features_path, 'name': 'higgs/train.csv'},
    "weights": {'path': features_path, 'name': 'higgs/weights.csv'},
    'test': {'path': features_path, 'name': 'higgs/test.csv'}
}
parameters = {
    'index': 'EventId',
    'features': [
        'DER_mass_MMC', 'DER_mass_transverse_met_lep', 'DER_mass_vis', 'DER_pt_h', 'DER_deltaeta_jet_jet',
        'DER_mass_jet_jet', 'DER_prodeta_jet_jet', 'DER_deltar_tau_lep', 'DER_pt_tot', 'DER_sum_pt',
        'DER_pt_ratio_lep_tau', 'DER_met_phi_centrality', 'DER_lep_eta_centrality', 'PRI_tau_pt', 'PRI_tau_pt',
        'PRI_tau_eta', 'PRI_tau_phi', 'PRI_lep_pt', 'PRI_lep_eta', 'PRI_lep_phi', 'PRI_met', 'PRI_met_phi',
        'PRI_met_sumet', 'PRI_jet_num', 'PRI_jet_leading_pt', 'PRI_jet_leading_eta', 'PRI_jet_leading_phi',
        'PRI_jet_subleading_pt', 'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi', 'PRI_jet_all_pt'
    ],
    'weights': ['Weight'],
    'cat': ['PRI_jet_num'],
    'target': ['Label'],
    'cols_out': [
        'DER_mass_MMC', 'DER_mass_transverse_met_lep', 'DER_mass_vis', 'DER_pt_h', 'DER_deltaeta_jet_jet',
        'DER_mass_jet_jet', 'DER_prodeta_jet_jet', 'DER_deltar_tau_lep', 'DER_pt_tot', 'DER_sum_pt',
        'DER_pt_ratio_lep_tau', 'DER_met_phi_centrality', 'PRI_tau_pt', 'PRI_tau_pt', 'PRI_tau_eta', 'PRI_tau_phi',
        'PRI_lep_pt', 'PRI_lep_eta', 'PRI_lep_phi', 'PRI_met', 'PRI_met_phi', 'PRI_met_sumet', 'PRI_jet_leading_pt',
        'PRI_jet_leading_eta', 'PRI_jet_subleading_eta', 'PRI_jet_all_pt', 'PRI_jet_num'
    ]
}


# Load and save dataset
l_cols = parameters['features'] + parameters['target']

df_train = pd.read_csv(
    os.path.join(inputs['train']['path'], inputs['train']['name']), header=0, index_col=parameters['index'],
)

df_test = pd.read_csv(
    os.path.join(inputs['test']['path'], inputs['test']['name']), header=0, index_col=parameters['index'],
    usecols=parameters['features'] + [parameters['index']]
)

# Prepare data
df_weights = df_train[parameters['weights']]
df_train, scaler = prepare_higgs_data(
    df_train[parameters['features'] + parameters['target']],
    l_col_cats=[parameters['cat']],
    l_targets=parameters['target'],
    missing_value=-999,
)
df_test, _ = prepare_higgs_data(
    df_test[parameters['features']],
    l_col_cats=[parameters['cat']],
    l_targets=parameters['target'],
    missing_value=-999,
    scaler=scaler
)

# minor fix (target name + col selection)
df_train.rename(columns={parameters['target'][0]: 'target'}, inplace=True)
df_train = df_train[parameters['cols_out'] + ['target']]
df_test = df_test[parameters['cols_out']]

# Save features
df_train.to_csv(os.path.join(outputs['train']['path'], outputs['train']['name']))
df_test.to_csv(os.path.join(outputs['test']['path'], outputs['test']['name']))
df_weights.to_csv(os.path.join(outputs['weights']['path'], outputs['weights']['name']))



