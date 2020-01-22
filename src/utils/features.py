# Global imports
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Local import


class FoldManager(object):
    """
    FoldManager manage segmentation of data for cross validation.

    """
    allowed_methods = ['standard', 'stratified']

    def __init__(self, df, params_builder, nb_folds, method='standard', test_size=0.2, target_name='target'):

        self.target_name = target_name

        # Split data set into a train / test
        self.df_data, self.df_test = train_test_split(df, test_size=test_size, shuffle=True)

        # Set method to build feature
        self.params_builder = params_builder

        # Set sampling method for Kfold
        if method == 'standard':
            self.kf = KFold(n_splits=nb_folds, shuffle=True)

        elif method == 'stratified':
            self.kf = StratifiedKFold(n_splits=nb_folds)

        else:
            raise ValueError('Choose method from {}'.format(FoldManager.allowed_methods))

        self.feature_builder = None

    def reset(self):
        self.feature_builder = None

    def get_all_features(self, params_features):
        """
        Build a data set composed of models. The target is also return, if specified.

        Parameters
        ----------
        params_features : dict
            kw parameters to build features.

        Returns
        -------
        tuple
            Features and target as numpy.ndarray

        """
        # Create models builder if necessary
        if self.feature_builder is None:
            self.feature_builder = FeatureBuilder(**self.params_builder)\
                .build(self.df_data, params_features)

        X, y = self.feature_builder.transform(self.df_data, target=True)

        return X, y

    def get_test_features(self, params_features):
        """
        Build test data set composed of models and transformed target.

        Parameters
        ----------
        params_features : dict
            kw parameters to build features.

        Returns
        -------
        tuple
            Features and target as numpy.ndarray

        """
        # Create models builder if necessary
        if self.feature_builder is None:
            self.feature_builder = FeatureBuilder(**self.params_builder)\
                .build(self.df_data, params_features)

        X, y = self.feature_builder.transform(self.df_test, target=True)

        return X, y

    def get_features(self, df):
        """
        Build feature.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame to transform  into feature.

        Returns
        -------
        tuple
            Features as numpy.ndarray

        """
        assert self.feature_builder is not None, "Feature builder is None"
        return self.feature_builder.transform(df)

    def generate_folds(self, params_features):
        """
        Generate train and validation data set. A data set is composed of models and target.

        Parameters
        ----------
        params_features : dict
            kw parameters to build features.

        Returns
        -------
        tuple
            Composed of dict with Features and target, for train and validation etl.
            i.e: {'X': numpy.ndarray, 'y': numpy.ndarray}
        """
        # Iterate over different folds
        for l_train, l_val in self.kf.split(self.df_data):

            # Create models  builder if necessary
            if self.feature_builder is None:
                self.feature_builder = FeatureBuilder(**self.params_builder). \
                    build(self.df_data.loc[self.df_data.index[l_train]], params_features)

            # Get features
            X, y = self.feature_builder.transform(self.df_data, target=True)

            # Build train / validation set
            X_train, y_train = X[l_train, :], y[l_train]
            X_val, y_val = X[l_val, :], y[l_val]

            yield {'X': X_train, 'y': y_train}, {'X': X_val, 'y': y_val}


class FeatureBuilder(object):
    """
    The FeatureBuilder manage the transformation of processed data composed of job description labelled by normalized
    positions. Its transformation pipeline is composed of:
    """

    def __init__(self, method=None, cat_cols=None, target_name='target'):
        """

        Attributes
        ----------
        method : str
            Model to use to transform text data.

        token_delimiter : str
            Delimiter that is used to seperate token in text input.
        """
        self.method, self.target_name, self.cat_cols = method, target_name, cat_cols
        self.model, self.label_encoder, self.is_built = None, None, None

    def build(self, df_data=None, params=None, force_train=False):
        """
        Build models from processed data and perform a numerical transform of label. The processed data is composed,
        on one hand, of text description, that will transform to numerical vector using TF-ID. On the other hand of a
        text label that will be numerised using one hot encoding.

        Parameters
        ----------
        df_data : pandas.DataFrame
            DataFrame of processed data.
        params :  dict
            Kw params of the TF-IDF.
        force_train : bool
            whether to fit TF-IDF on the data.

        Returns
        -------
        self : Current instance of the class.
        """

        # Create and fit the TF-IDF if necessary
        if self.model is None or force_train:

            if self.method == 'cat_encode':
                self.model = OneHotEncoder(
                    handle_unknown='ignore', sparse=params.get('sparse', False), dtype=params.get('dtype', bool)
                )
                self.model.fit(df_data[self.cat_cols])

        self.is_built = True

        return self

    def transform(self, df_data, target=False):
        """
        Perform a numerical transform of the text in DataFrame df, using previously fitted TF-IDF.
        In addition, perform a numerical transform of target if specified, using a multiple class hot encoding.

        Parameters
        ----------
        df_data : pandas.DataFrame
            DataFrame of processed data that shall be transformed.

        target : bool
            Whether to transform target or not.

        Returns
        -------
        numpy.array
            Transformed data.

        """

        if not self.is_built:
            raise ValueError("Transforming data requires building features.")

        if self.method == 'cat_encode':
            X = self.model.transform(df_data[self.cat_cols])
            X = np.hstack((
                X, df_data[[c for c in df_data.columns if c not in self.cat_cols and c != self.target_name]]
            ))

        else:
            X = df_data[[c for c in df_data.columns if c != self.target_name]].values

        if target:
            return X, df_data.loc[:, self.target_name].values

        return X
