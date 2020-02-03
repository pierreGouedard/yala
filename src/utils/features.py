# Global imports
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import numpy as np
from scipy.sparse import csc_matrix
from sklearn.preprocessing import LabelEncoder

# Local import
from src.tools.encoder import HybridEncoder, CatEncoder


class FoldManager(object):
    """
    FoldManager manage segmentation of data for cross validation.

    """
    allowed_methods = ['standard', 'stratified']

    def __init__(self, df, params_builder, nb_folds, method='standard', test_size=0.2, target_name='target'):

        self.target_name = target_name

        # Split data set into a train / test
        self.df_train, self.df_test = train_test_split(df, test_size=test_size, shuffle=True)

        # Set method to build feature
        self.params_builder = params_builder

        # Set sampling method for Kfold
        if nb_folds <= 2:
            self.kf = None

        elif method == 'standard':
            self.kf = KFold(n_splits=nb_folds, shuffle=True)

        elif method == 'stratified':
            self.kf = StratifiedKFold(n_splits=nb_folds)

        else:
            raise ValueError('Choose Kfold method from {}'.format(FoldManager.allowed_methods))

        self.feature_builder = None

    def reset(self):
        self.feature_builder = None

    def get_train_data(self, params_features, force_recompute=False):
        """
        Build a data set composed of models. The target is also return, if specified.

        Parameters
        ----------
        params_features : dict
            kw parameters to build features.

        force_recompute : bool
            If True, it fit feature builder with train data

        Returns
        -------
        tuple
            Features and target as numpy.ndarray

        """
        # Create models builder if necessary
        if self.feature_builder is None or force_recompute:
            self.feature_builder = FeatureBuilder(**self.params_builder)\
                .build(self.df_train, params_features)

        X, y = self.feature_builder.transform(self.df_train, target=True)

        return X, y

    def get_test_data(self, params_features=None):
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
                .build(self.df_train, params_features)

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

        if self.kf is not None:
            # Iterate over different folds
            for l_train, l_val in self.kf.split(self.df_train):

                # Create models  builder if necessary
                self.feature_builder = FeatureBuilder(**self.params_builder)\
                    .build(self.df_train.loc[self.df_train.index[l_train]], params_features)

                # Get features
                X, y = self.feature_builder.transform(self.df_train, target=True)

                # Build train / validation set
                X_train, y_train = X[l_train, :], y[l_train]
                X_val, y_val = X[l_val, :], y[l_val]

                yield {'X': X_train, 'y': y_train}, {'X': X_val, 'y': y_val}

        else:

            X_train, y_train = self.get_train_data(params_features, force_recompute=True)
            X_test, y_test = self.get_test_data()
            yield {'X': X_train, 'y': y_train}, {'X': X_test, 'y': y_test}


class FeatureBuilder(object):
    """
    The FeatureBuilder manage the transformation of processed data composed of job description labelled by normalized
    positions. Its transformation pipeline is composed of:
    """

    def __init__(self, method=None, cat_cols=None, num_cols=None, target_name='target', target_transform=None):
        """

        Attributes
        ----------
        method : str
            Model to use to transform text data.

        token_delimiter : str
            Delimiter that is used to seperate token in text input.
        """
        self.method, self.target_name, self.cat_cols, self.num_cols = method, target_name, cat_cols, num_cols
        self.target_transform = target_transform

        self.model, self.target_encoder, self.is_built = None, None, None

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
                self.model = CatEncoder(
                    handle_unknown='ignore', sparse=params.get('sparse', False), dtype=params.get('dtype', bool)
                )
                self.model.fit(df_data[self.cat_cols])

            elif self.method == 'cat_num_encode':
                self.model = HybridEncoder(num_cols=self.num_cols, cat_cols=self.cat_cols,  **params)
                self.model.fit(df_data[self.cat_cols + self.num_cols])

            else:
                raise ValueError('Method not implemented: {}'.format(self.method))

            if self.target_transform == 'encoding':
                self.target_encoder = LabelEncoder().fit(df_data[self.target_name])

            elif self.target_transform == 'sparse_encoding':
                self.target_encoder = LabelEncoder().fit(df_data[self.target_name])

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
    if isinstance(y, spmatrix):
        if y.shape[1] == 1:
            y = y.toarray()[:, 0]

        else:
            y = y.toarray().argmax(axis=1)
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

        elif self.method == 'cat_num_encode':
            X = self.model.transform(df_data)

        else:
            X = df_data[[c for c in df_data.columns if c != self.target_name]].values

        if target:
            if self.target_transform == 'encoding':
                y = self.target_encoder.transform(df_data[self.target_name])

            elif self.target_transform == 'sparse_encoding':
                y = self.target_encoder.transform(df_data[self.target_name])
                if max(y) > 1:
                    y = csc_matrix(([True] * y.shape[0], (range(y.shape[0]), y)), shape=(y.shape[0], len(np.unique(y))))

                else:
                    y = csc_matrix(y[:, np.newaxis] > 0)

            else:
                y = df_data.loc[:, self.target_name].values

            return X, y

        return X
