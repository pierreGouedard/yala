import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, hstack, csr_matrix, coo_matrix, lil_matrix
from sklearn.preprocessing import OneHotEncoder


class NumEncoder(object):
    """
    NumEncoder build a discrete space from continuous numerical 2D arrays and encode numerical values of the array based
    on this discrete space. The Class implement the  BaseEstimator and TransformerMixin (or _BaseEncoder) interface
    from scikit-learn

    """
    bins_method = ['bounds', 'signal', 'quantile', 'clusters']

    def __init__(self, n_bins, method='bounds', n_quantile=10, n_cluster=10, bounds=None):
        """

        :param n_bins: Number of discrete values
        :type n_bins: int
        :param method: string specifying methods to discretize space spanned by numerical values in X
        :type method: str
        :param n_quantile: Number of quantile to take into account for quantile based method
        :type n_quantile: int
        :param n_cluster: Number of quantile to take into account for quantile based method
        :type n_cluster: int
        :param bounds: Upper and lower bounds of dicrete values.
        :type bounds: list
        """
        # Core parameters
        self.method = method
        self.n_bins = n_bins

        # Other parameters
        self.n_quantile = n_quantile
        self.n_cluster = n_cluster
        self.bounds = bounds

        # Set unknown attribute to None
        self.bins = None

    def fit_transform(self, X, y=None):
        """
        Fit 2D array X and transform its values.

        :param X: 2D array of numerical values
        :type X: 2D numpy array
        :param y: Array of target class (not used)
        :return: Array of encoded numerical values
        :rtype: 2D numpy array

        """
        self.fit(X, y)
        return self.transform(X)

    def fit(self, X, y=None):
        """
        Build a set of discrete values from numerical value of the 2D array X.

        :param X: 2D array of numerical values
        :type X: 2D numpy array
        :param y: Array of target class (not used)
        :return: Current instance of the class
        :rtype: self

        """
        if self.method == 'signal':
            self.bounds = np.quantile(X[~np.isnan(X)], [0.05, 0.95])
            self.n_bins = min(len(np.unique(X)), self.n_bins)

        self.bins = np.linspace(self.bounds[0], self.bounds[1], num=self.n_bins)

        return self

    def transform(self, ax_continuous):
        ax_activation = abs(self.bins - ax_continuous)
        ax_activation = ax_activation == ax_activation.min(axis=1, keepdims=True)
        return csc_matrix(ax_activation)

    def inverse_transform(self, sax_bits, agg='mean'):
        raise NotImplementedError

    def discretize_value(self, x):

        if self.bins is None:
            raise ValueError('First set the bins of discretizer')

        x_ = min([(v, abs(x - v)) for k, v in self.bins.items()], key=lambda t: t[1])[0]

        return x_

    def discretize_array(self, ax_continuous):

        # Vectorize discretie value function
        vdicretizer = np.vectorize(lambda x: self.discretize_value(x))

        # Apply to array
        ax_discrete = vdicretizer(ax_continuous)

        return ax_discrete

    def arange(self, x_min, x_max):
        x_min_, x_max_ = self.discretize_value(x_min), self.discretize_value(x_max)
        return np.array(sorted([v for _, v in self.bins.items() if x_min_ <= v <= x_max_]))


class CatEncoder(OneHotEncoder):
    """
    CatEncoder build a one hot encoding of categorical feature. The Class inherit from OneHotEncoder class from
    scikit-learn. It is coded as it just in case we need some extra feature in a close future.

    """


class HybridEncoder():

    def __init__(self, cat_cols, num_cols, params_num_enc, params_cat_enc):
        """

        :param cat_cols:
        :param num_cols:
        :param params_num_enc:
        :param params_cat_enc:
        """

        # Save cat and numerical columns name or indices
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.sax_feature_to_input = None

        # Create encoders
        self.cat_enc = CatEncoder(**params_cat_enc)
        self.num_encs = {c: NumEncoder(**params_num_enc) for c in num_cols}
        self.n_bins_num = params_num_enc['n_bins']
        self.map_encoders = None

    def fit_transform(self, X, y=None):
        """
        Fit 2D array X and transform its values.

        :param X: 2D array of numerical values
        :type X: 2D numpy array
        :param y: Array of target class (not used)
        :return: Array of encoded numerical values
        :rtype: 2D numpy array

        """
        self.fit(X, y)
        return self.transform(X)

    def fit(self, X, y=None):
        """
        Build a set of discrete values from numerical value of the 2D array X.

        :param X: 2D array of numerical values
        :type X: 2D numpy array
        :param y: Array of target class (not used)
        :return: Current instance of the class
        :rtype: self

        """
        # Fit categorical encoder
        self.cat_enc.fit(self.get_array_from_input(X, self.cat_cols))

        # Initialize feature to input mapping
        n_inputs = sum([len(l_cat) for l_cat in self.cat_enc.categories_]) + len(self.num_cols) * self.n_bins_num
        ax_feature_to_input = np.zeros((n_inputs, len(self.cat_cols) + len(self.num_cols)), dtype=bool)

        # fill feature to input mapping
        n = 0
        for i, l_cats in enumerate(self.cat_enc.categories_):
            ax_feature_to_input[range(n, n + len(l_cats)), i] = True
            n += len(l_cats)

        # Fit numerical encoders
        for i, c in enumerate(self.num_cols):
            self.num_encs[c].fit(self.get_array_from_input(X, [c]))
            ax_feature_to_input[range(n, n + self.n_bins_num), i + len(self.cat_cols)] = True
            n += self.n_bins_num

        self.sax_feature_to_input = csc_matrix(ax_feature_to_input)
        return self

    def transform(self, X):
        # Transform categorical features
        X_encoded = self.cat_enc.transform(self.get_array_from_input(X, self.cat_cols)).tocsc()

        # transform numerical features
        l_num_encoded, n = [], X_encoded.shape
        for c in self.num_cols:
            l_num_encoded.append(self.num_encs[c].transform(self.get_array_from_input(X, [c])))

        return hstack([X_encoded] + l_num_encoded)

    @staticmethod
    def get_array_from_input(X, l_indices):
        if isinstance(X, pd.DataFrame):
            return X[l_indices].values

        elif isinstance(X, np.ndarray):
            return X[:, l_indices]

        else:
            raise ValueError('Data type {} not understood'.format(type(X)))
