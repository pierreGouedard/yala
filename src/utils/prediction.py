# Global imports
import logging
import itertools
import pandas as pd
import numpy as np
import dill as pickle
import os
from sklearn.metrics import precision_score, accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, \
    confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
import xgboost
from scipy.sparse import spmatrix

# Local import
from src.utils.names import KVName
from src.utils.features import FoldManager
from src.model.yala import Yala

logging.getLogger().setLevel(logging.INFO)


class ClassifierSelector(object):
    """
    The ClassifierSelector is an object that supervise the selection of hyper parameter of models building and classifier
    in the scope of a multi target document classification.{"callbacks": [es]}

    """
    allowed_score = ['precision', 'accuracy', 'balanced_accuracy', 'f1_score', 'roc']
    allowed_model = ['dt', 'rf', 'xgb', 'yala']

    def __init__(self, df_data, model_classification, params_features, params_features_grid, params_mdl, params_mdl_grid,
                 params_fold, scoring, path_backup=None):
        """

        Parameters
        ----------
        df_corpus : pandas.DataFrame
            DataFrame composed of document text and label.

        model_classification : str
            Name of the model to use to classify documents.

        params_features : dict
            Dictionary of fixed parameter for building features.

        params_features_grid : dict
            Dictionary of parameter for building features to select using a grid search.

        params_mdl : dict
            Dictionary of fixed parameter for model.

        params_mdl_grid : dict
            Dictionary of parameter for model to select using a grid search.

        params_fold : dict
            Dictionary of parameter to use to make the cross validation of features builder and model.

        scoring : str
            Scoring method to use to evaluate models and models builder.

        """
        assert model_classification in self.allowed_model, "Choose model among {}".format(self.allowed_model)
        assert scoring in self.allowed_score, "Choose scoring among {}".format(self.allowed_score)

        # Core parameter classifier
        self.model_classification = model_classification
        self.scoring = scoring

        # Parameters of models builder and model
        self.params_features = params_features
        self.params_features_grid = params_features_grid
        self.params_mdl = params_mdl
        self.params_mdl_grid = params_mdl_grid

        # Core attribute of classifier
        self.fold_manager = FoldManager(df_data, **params_fold)
        self.is_fitted = False
        self.model = None
        self.d_search = None
        self.backup = path_backup

    def fit(self):
        """
        Find the optimal combination of hyper parameter of models building and model through grid search. Fit the
        entire etl with the selected parameters.

        """
        # Fit model and models builder using K-fold cross validation
        self.d_search = self.grid_search()
        self.model = self.__fit_all()
        self.is_fitted = True

        return self

    def grid_search(self):
        """
        Perform a cross validation of hyper parameters of the features building routine and the model used for
        classification. For given parameter of models builder, we retain the parameter of prediction model that get
        the best score. The parameter of models builder with the higher score will be chosen, along with the best model
        associated.

        """

        d_search, best_score = {}, 0.
        for i, cross_values in enumerate(itertools.product(*self.params_features_grid.values())):

            # Reset fold_manager to allow fit of new models builder
            self.fold_manager.reset()

            # Update params of grid search
            d_features_grid_instance = self.params_features.copy()
            d_features_grid_instance.update(dict(zip(self.params_features_grid.keys(), cross_values)))
            d_search[i] = {'params_feature': d_features_grid_instance.copy()}

            # Inform about Current params
            logging.info("[FEATURE]: {}".format(KVName.from_dict(d_features_grid_instance).to_string()))

            # Fit model and keep params of best model associated with current models's parameters
            best_mdl_params, best_mdl_score = self.grid_search_mdl(d_features_grid_instance)
            d_search[i].update({'params_mdl': best_mdl_params, 'best_score': best_mdl_score})

            # Keep track of best association models builder / model
            if best_score < best_mdl_score:
                d_search['best_key'] = i

            # Backup search
            if self.backup is not None:
                with open(self.backup, 'wb') as handle:
                    pickle.dump(d_search, handle)

        logging.info('Optimal parameters found are {}'.format(d_search[d_search['best_key']]))

        return d_search

    def grid_search_mdl(self, d_feature_params):
        """
        Perform a cross validation of hyper parameter of the logistic regression for given word embedding parameter.

        Parameters
        ----------
        d_feature_params : dict
            Dictionary of parameters for building features.

        Returns
        -------
        tuple
            A tuple containing a dictionary of parameters of the best model and the score of the best model
        """

        best_mdl_params, best_mdl_score = None, 0.
        for cross_values in itertools.product(*self.params_mdl_grid.values()):

            # Update params of logistic regression model
            d_mdl_grid_instance = self.params_mdl.copy()
            d_mdl_grid_instance.update(dict(zip(self.params_mdl_grid.keys(), cross_values)))

            # Instantiate model and cross validate the hyper parameter
            model, args = get_model(self.model_classification, d_mdl_grid_instance)
            l_errors = []

            for d_train, d_test in self.fold_manager.generate_folds(d_feature_params):
                # Fit and predict
                if 'input_shape' in args.keys():
                    args['input_shape'] = d_train['X'].shape[1]

                model.fit(d_train['X'], d_train['y'], **args)
                score = get_score(self.scoring, model.predict(d_test['X']), d_test['y'])

                # Get error
                l_errors.append(score)

            # Average errors and update best params if necessary
            mu = np.mean(l_errors)
            logging.info('[MODEL]: {} | score: {}'.format(KVName.from_dict(d_mdl_grid_instance).to_string(), mu))

            if best_mdl_score < mu:
                best_mdl_params, best_mdl_score = d_mdl_grid_instance.copy(), mu

        return best_mdl_params, best_mdl_score

    def __fit_all(self):
        """
        Fit models builder and model based on optimal hyper parameter found using.

        :return: trained model

        """

        # Reset data_manager to allow fit of we
        self.fold_manager.reset()

        # Recover optimal parameters
        d_feature_params = self.d_search[self.d_search['best_key']]['params_feature']
        d_model_params = self.d_search[self.d_search['best_key']]['params_mdl']

        X, y = self.fold_manager.get_all_features(d_feature_params)

        # Instantiate model and fit it
        model, args = get_model(self.model_classification, d_model_params)

        if 'input_shape' in args.keys():
            args['input_shape'] = X.shape[1]

        model.fit(X, y, **args)

        return model

    def save_classifier(self, path):
        """
        Save core element of the documetn classifier.

        Parameters
        ----------
        path : str
            path toward the location where classifier shall be saved.

        Returns
        -------

        """

        d_param_feature = self.d_search[self.d_search['best_key']]['params_feature']
        d_param_model = self.d_search[self.d_search['best_key']]['params_mdl']

        # Instantiate classifier
        document_classifier = Classifier(
            self.model, self.fold_manager.feature_builder, d_param_model, d_param_feature
        )

        # Pickle classifier
        with open(path, 'wb') as handle:
            pickle.dump(document_classifier, handle)

        return self

    def save_data(self, path, name_train, name_test):
        """
        Save data used to select and fit the document classifier.

        Parameters
        ----------
        path : str
            path toward the location where classifier shall be saved.
        name_train : str
            name of file containing train data.
        name_test : str
            name of file containing test data.

        Returns
        -------

        """

        path_train, path_test = os.path.join(path, name_train), os.path.join(path, name_test)
        self.fold_manager.df_data.to_hdf(path_train, key=name_train.split('.')[0], mode='w')
        self.fold_manager.df_test.to_hdf(path_test, key=name_test.split('.')[0], mode='w')

        return self


class Classifier(object):
    """
    The Classifier is an object that ready to use to classify text document.

    """

    def __init__(self, model_classification, feature_builder, params_model, params_features):
        """

        Parameters
        ----------
        model_classification : object
            Fitted model to use to classify documents.

        feature_builder : src.model.feature.FeatureBuilder
            Fitted model to use to vectorize text of documents.

        params_features : dict
            Dictionary of fixed parameter for building features.

        params_model : dict
            Dictionary of fixed parameter for classification model.


        """

        # Core parameter classifier
        self.model_classification = model_classification
        self.feature_builder = feature_builder
        self.params_model = params_model
        self.params_features = params_features

    @staticmethod
    def from_path(path):
        with open(path, 'rb') as handle:
            dc = pickle.load(handle)

        return Classifier(**dc.__dict__)

    def predict(self, df):
        """
        Predict target for feature in df.

        Parameters
        ----------
        df : pandas.DataFrame

        Returns
        -------

        """
        features = self.feature_builder.transform(df)
        preds = self.model_classification.predict(features)

        return pd.Series(preds, index=df.index, name="prediction")

    def predict_proba(self, df):
        """
        Predict probabilities over target space for feature in df.

        Parameters
        ----------
        df : pandas.DataFrame

        Returns
        -------

        """
        features = self.feature_builder.transform(df)
        preds = self.model_classification.predict_proba(features)
        return pd.DataFrame(preds, index=df.index, columns=self.feature_builder.label_encoder.classes_)

    def evaluate(self, df_train, df_test):
        """
        Compute different metrics for evaluation of the current classifier performance using labelled data set.

        Parameters
        ----------
        df_train : pandas.DataFrame
            Corpus of labelled document to evaluate the classifier.

        df_test : pandas.DataFrame
            Corpus of labelled document to evaluate the classifier.

        Returns
        -------

        """

        # Build features
        X_train, y_train = self.feature_builder.transform(df_train, target=True)
        X_test, y_test = self.feature_builder.transform(df_test, target=True)

        # Instantiate model and fit it
        yhat_train = self.model_classification.predict(X_train)
        yhat_test = self.model_classification.predict(X_test)

        # Compute confusion matrix
        confmat_train = confusion_matrix(yhat_train, y_train)
        confmat_test = confusion_matrix(yhat_test, y_test)

        # Compute multiple score
        d_scores = {'train': compute_scores(yhat_train, y_train, self.feature_builder.label_encoder)}
        d_scores.update({'test': compute_scores(yhat_test, y_test, self.feature_builder.label_encoder)})

        return confmat_train, confmat_test, d_scores


def get_model(model_name, model_params):
    """

    Parameters
    ----------
    model_name
    model_params

    Returns
    -------

    """

    if model_name == 'rf':
        return RandomForestClassifier(**model_params), {}

    elif model_name == 'dt':
        return DecisionTreeClassifier(**model_params), {}

    elif model_name == 'xgb':
        return xgboost.XGBClassifier(**model_params), {}

    elif model_name == 'yala':
        return Yala(**model_params), {}

    else:
        raise ValueError('Model name not understood: {}'.format(model_name))


def get_score(scoring, yhat, y, average='macro', labels=None):

    if isinstance(y, spmatrix):
        y = y.toarray().argmax(axis=1)

    if scoring == 'precision':
        if labels is None and average is not None:
            labels = list(set(y).intersection(set(yhat)))

        score = precision_score(y, yhat, labels=labels, average=average)

    elif scoring == 'accuracy':
        score = accuracy_score(y, yhat)
        import IPython
        IPython.embed()

    elif scoring == 'balanced_accuracy':
        score = balanced_accuracy_score(y, yhat)

    elif scoring == 'f1_score':
        if labels is None and average is not None:
            labels = list(set(y).intersection(set(yhat)))

        score = f1_score(y, yhat, labels=labels, average=average)

    elif scoring == 'roc':
        # One hot encode the targets
        enc = OneHotEncoder(categories='auto').fit(y[:, np.newaxis])

        # Compute score
        score = roc_auc_score(
            enc.transform(y[:, np.newaxis]).toarray(), enc.transform(yhat[:, np.newaxis]).toarray(), average=average
        )

    else:
        raise ValueError('Scoring name not understood: {}'.format(scoring))

    return score


def compute_scores(yhat, y, label_encoder):

    # Init score dict
    d_scores = {}

    # Compute precision for each predicted label
    l_predicted_labels = list(set(yhat).intersection(y))
    l_precisions = get_score('precision', yhat, y, average=None, labels=l_predicted_labels)
    d_scores['precision'] = {label_encoder.classes_[l_predicted_labels[i]]: p for i, p in enumerate(l_precisions)}
    d_scores['label_unpredicted'] = [label_encoder.classes_[i] for i in set(y).difference(yhat)]

    # Compute other global metrics
    d_scores['accuracy'] = get_score('accuracy', yhat, y)
    d_scores['balanced_accuracy'] = get_score('balanced_accuracy', yhat, y)
    d_scores['f1_score'] = get_score('f1_score', yhat, y, average='micro')

    # Compute roc_auc for each predicted label
    l_rocs = get_score('roc', yhat, y, average=None)
    d_scores['roc_auc'] = {label_encoder.classes_[i]: p for i, p in enumerate(l_rocs)}

    return d_scores