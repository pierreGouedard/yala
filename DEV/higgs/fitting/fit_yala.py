# Global import
import os
import pandas as pd

# Local import
from settings import models_path, features_path, export_path
from src.dev.names import KVName
from src.tools.ams_metric import embedded_ams
from src.dev.prediction import ClassifierSelector
from src.tools.ams_metric import AMS_metric

# Declare input and outputs
inputs = {
    'train': {'path': features_path, 'name': 'higgs/train.csv'},
    'weights': {'path': features_path, 'name': 'higgs/weights.csv'}
}
outputs = {
    'model': {'path': models_path, 'name': 'higgs/{}.pickle'},
    'solution': {'path': export_path, 'name': 'higgs/solution_{}.csv'},
    'submission': {'path': export_path, 'name': 'higgs/submission_{}.csv'},
}

parameters = {
    'model': 'yala'
}

# Set params for cross validation
params_folds = {
    'nb_folds': 2,
    'method': 'standard',
    'test_size': 0.2,
    'params_builder': {
        'method': 'cat_num_encode',
        'num_cols': [
            'DER_mass_MMC', 'DER_mass_transverse_met_lep', 'DER_mass_vis', 'DER_pt_h', 'DER_deltaeta_jet_jet',
            'DER_mass_jet_jet', 'DER_prodeta_jet_jet', 'DER_deltar_tau_lep', 'DER_pt_tot', 'DER_sum_pt',
            'DER_pt_ratio_lep_tau', 'DER_met_phi_centrality', 'DER_lep_eta_centrality', 'PRI_tau_pt', 'PRI_tau_pt',
            'PRI_tau_eta', 'PRI_tau_phi', 'PRI_lep_pt', 'PRI_lep_eta', 'PRI_lep_phi', 'PRI_met', 'PRI_met_phi',
            'PRI_met_sumet', 'PRI_jet_leading_pt', 'PRI_jet_leading_eta', 'PRI_jet_leading_phi',
            'PRI_jet_subleading_pt', 'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi', 'PRI_jet_all_pt'
        ],
        'cat_cols': ['PRI_jet_num'],
        'target_transform': 'sparse_encoding'
    }
}

params_yala = {
    'sampling_rate': 0.5, 'n_sampling': 1, 'max_iter': 100, 'learning_rate': 1e-2, 'batch_size': 150000,
    'drainer_batch_size': 30000, 'min_firing': 1000, 'min_precision': 0.8, 'max_precision': 0.99, 'max_retry': 5,
    'dropout_vertex': 0.05, 'dropout_mask': 0.3
}
params_yala_grid = {}
params_encoding = {
    'params_num_enc': {'n_bins': 20, 'method': 'signal'},
    'params_cat_enc': {'sparse': True, 'dtype': bool, 'handle_unknown': 'ignore'},
}

# Defined custom output name based on parameters of the script
name_mdl = '{}'.format(KVName.from_dict(parameters).to_string())
model_path = os.path.join(outputs['model']['path'], outputs['model']['name']).format(name_mdl)
submission_path = os.path.join(outputs['submission']['path'], outputs['submission']['name']).format(name_mdl)
solution_path = os.path.join(outputs['solution']['path'], outputs['solution']['name']).format(name_mdl)

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

# display score
_, _, d_scores = classifier.evaluate(cs.fold_manager.df_train, cs.fold_manager.df_test)
print(d_scores)
print(params_yala)

# Save submission file
df_submission = classifier.predict(cs.fold_manager.df_test)\
    .to_frame('Class')\
    .assign(
        Class=lambda x: classifier.feature_builder.target_encoder.inverse_transform(x.Class),
        RankOrder=lambda x: range(len(x))
    )\
    .reset_index()\
    .loc[:, ['EventId', 'RankOrder', 'Class']]\
    .to_csv(submission_path, index=None)

# Save Solution file
df_solution = pd.merge(cs.fold_manager.df_test['target'], df_weights, left_index=True, right_index=True, how='left')\
    .rename(columns={'target': 'Class'})\
    .reset_index()\
    .loc[:, ['EventId', 'Class', 'Weight']] \
    .to_csv(solution_path, index=None)

AMS_metric(solution_path, submission_path)
