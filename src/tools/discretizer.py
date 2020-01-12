import numpy as np
from scipy.sparse import csc_matrix


# SHOULD IMPLEMENT THE ABSTRACT CLASS OF IMPUTER FROM SCIKIT-LEARN
# It should have minimum attributes:
#
# * parameters
#
# It should have minimum method
#
# * fit(self, X, y=None)
# * fit_transform(self, X, y=None)
# * transform(self, X)
# X should be 2 d array (n_sample, n_features)

class NumDiscretizer:

    bins_method = ['bounds', 'signal', 'quantiles', 'clusters']

    def __init__(self, n_bins, method='bins', n_quantile=10, n_cluster=10, bounds=None):

        # Core parameters
        self.method = method
        self.n_bins = n_bins

        # Other parameters
        self.n_quantile=n_quantile
        self.n_cluster = n_cluster
        self.bounds = bounds

        # Set unknown attribute to None
        self.bins = None

    def fit(self, X, y=None, method='bounds'):

        if method == 'bounds':
            self.bins = {i: x for i, x in enumerate(np.linspace(self.bounds[0], self.bounds[1], self.n_bins))}

        elif method == 'signal':
            max_, min_ = max(X), min(X)
            self.fit_from_bound(max_, min_)

        elif method == 'quantiles':
            n = self.n_quantile
            l_quantiles = [np.percentile(s, 100 * (float(i) / n)) for i in range(n + 1)]
            self.fit_from_quantiles(l_quantiles, self.n_bins / n)

        elif method == 'clusters':
            n = self.n_cluster
            raise NotImplementedError

        else:
            raise ValueError(
                'Method to compute bins not understood: {}. choose from {}'.format(method, self.bins_method)
            )

        return self

    def fit_from_bound(self, max_, min_, res=1e-4):

        # Computes bins
        if max_ - res < min_:
            max_ = min_

        self.bins = {i: x for i, x in enumerate(np.unique(np.linspace(min_, max_, self.n_bins)))}

    def fit_from_quantiles(self, l_quantiles, k):

        self.bins = {}
        for i, (min_, max_) in enumerate(zip(l_quantiles[:-1], l_quantiles[1:])):
            self.bins.update({i * k + j: x for j, x in enumerate(np.unique(np.linspace(min_, max_, k + 1))[:-1])})

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

    def transform(self, ax_continuous):

        ax_bits = np.zeros((ax_continuous.shape[0], ax_continuous.shape[1] * 2 * self.n_bins), dtype=bool)
        ax_discrete = self.discretize_array(ax_continuous)

        for i, ax in enumerate(ax_discrete):
            for j, x in enumerate(ax):

                ax_code = np.array(
                    [b <= x for _, b in sorted(self.bins.items(), key=lambda t: t[0])] +
                    [b > x for _, b in sorted(self.bins.items(), key=lambda t: t[0])],
                    dtype=bool
                )
                ax_bits[i, j * 2 * self.n_bins: (j + 1) * 2 * self.n_bins] = ax_code

        return csc_matrix(ax_bits)

    # TODO: Make it 2D
    def inverse_transform(self, sax_bits):

        ax_bits = sax_bits.toarray()[0]
        ax_discrete = np.zeros(ax_bits.shape[0], int(ax_bits.shape[1] / (2 * self.n_bins)))

        for i in range(ax_discrete.shape[0]):
            for j in range(ax_discrete.shape[1]):
                # TODO: where may not the correct thing to do. From all activated values take the largest one and set ax_discrete
                l_values = np.where(ax_bits[j * 2 * self.n_bins: ((j * 2) + 1) * self.n_bins])
                ax_discrete[i, j] = max(l_values, key='pute')

        return ax_discrete

    def arange(self, x_min, x_max):
        x_min_, x_max_ = self.discretize_value(x_min), self.discretize_value(x_max)
        return np.array(sorted([v for _, v in self.bins.items() if x_min_ <= v <= x_max_]))

