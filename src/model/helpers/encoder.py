# Local import
import numpy as np
from scipy.sparse import csc_matrix, hstack

# Global import


class Encoder(object):
    def __init__(self, n_bin, bin_method='uniform', bin_missing=False, min_val_by_bin=10):

        # Core parameters
        self.n_bin, self.bin_method, self.bin_missing = n_bin, bin_method, bin_missing
        self.min_val_by_bin = min_val_by_bin

        self.total_size = self.n_bin + self.bin_missing

        # Set unknown attribute to None
        self.bins = None

    def update_total_size(self):
        self.total_size = self.n_bin + self.bin_missing

    def fit(self, x, y=None):

        # Get unique values
        ax_unique = np.unique(x)

        # If not enough unique values, treat it as a cat value and hot encode it
        if len(ax_unique) < (self.n_bin * self.min_val_by_bin):
            self.bins = ax_unique
            self.n_bin = len(ax_unique)
            self.update_total_size()

            return self

        # Encode numerical values
        if self.bin_method == 'uniform':
            bounds = np.quantile(x[~np.isnan(x)], [0.02, 0.98])
            self.bins = np.linspace(bounds[0], bounds[1], num=self.n_bin)

        elif self.bin_method == 'quantile':
            # Get bins from quantiles
            ax_bins = np.quantile(x[~np.isnan(x)], [i / (self.n_bin + 1) for i in range(1, self.n_bin + 1)])

            # Get rid of too close bounds
            ax_filter = np.hstack([abs(ax_bins[:-1] - ax_bins[1:]) >= 1e-4, np.array([True])])
            self.bins = ax_bins[ax_filter] + np.random.rand(ax_filter.sum()) * 1e-4

            # Update Number of bin
            self.n_bin = len(self.bins)
            self.update_total_size()

        else:
            raise ValueError(f"Unknown bin method {self.bin_method}")

        return self

    def transform(self, ax_continuous):

        if self.bins is None:
            raise ValueError('Bins are not set when transform called')

        ax_activation = abs(self.bins - ax_continuous)
        ax_activation = ax_activation == ax_activation.min(axis=1, keepdims=True)

        if self.bin_missing:
            ax_activation = np.hstack([ax_activation, ~ax_activation.any(axis=1, keepdims=True)])

        return csc_matrix(ax_activation)


class MultiEncoders():
    _NUMERIC_KINDS = set('buifc')

    def __init__(self, n_bin, bin_method, bin_missing=False):
        # global parameters
        self.p_encoder = {"n_bin": n_bin, "bin_method": bin_method, "bin_missing": bin_missing}

        # parameters to fit
        self.basis = None
        self.bf_map = None
        self.n_label = None
        self.encoders = {}

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X, y)

    def fit(self, X, y):

        # Check whether X and y contain only numeric data
        assert X.dtype.kind in self._NUMERIC_KINDS, "X contains non num data, must contains only num data"
        assert y.dtype.kind in self._NUMERIC_KINDS, "y contains non num data, must contains only num data"

        if self.basis is not None:
            pass

        ax_bf_map, n_inputs = np.zeros((self.p_encoder['n_bin'] * X.shape[1], X.shape[1]), dtype=bool), 0
        for i in range(X.shape[1]):
            self.encoders[i] = Encoder(**self.p_encoder).fit(X[:, i])
            ax_bf_map[range(n_inputs, n_inputs + self.encoders[i].total_size), i] = True
            n_inputs += self.encoders[i].total_size

        self.bf_map = csc_matrix(ax_bf_map[:n_inputs, :])
        self.n_label = len(np.unique(y))

        return self

    def transform(self, X, y):
        assert self.bf_map is not None, "Encoder is not fitted when transform called"

        # Transform target
        if self.n_label > 1:
            y = csc_matrix(([True] * y.shape[0], (range(y.shape[0]), y)), shape=(y.shape[0], self.n_label))

        else:
            y = csc_matrix(y[:, np.newaxis] > 0)

        # Transform features
        if self.basis is not None:
            pass

        # Transform inputs
        l_encoded = []
        for i in range(X.shape[1]):
            l_encoded.append(self.encoders[i].transform(X[:, [i]]))

        return hstack(l_encoded), y
