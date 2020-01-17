import numpy as np
from scipy.sparse import csc_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


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

        if self.method == 'bounds':
            self.bins = {i: x for i, x in enumerate(np.linspace(self.bounds[0], self.bounds[1], self.n_bins))}

        elif self.method == 'signal':
            max_, min_ = max(X), min(X)
            self.fit_from_bound(max_, min_)

        elif self.method == 'quantile':
            n = self.n_quantile
            l_quantiles = [np.percentile(X, 100 * (float(i) / n)) for i in range(n + 1)]
            self.fit_from_quantiles(l_quantiles, self.n_bins / n)

        elif self.method == 'clusters':
            n = self.n_cluster
            raise NotImplementedError

        else:
            raise ValueError(
                'Method to compute bins not understood: {}. choose from {}'.format(self.method, self.bins_method)
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

        # Set last quantile to max value
        self.bins[max(self.bins.keys())] = l_quantiles[-1]

    def transform(self, ax_continuous):

        ax_bits = np.zeros((ax_continuous.shape[0], ax_continuous.shape[1] * 2 * self.n_bins), dtype=bool)

        for i, ax in enumerate(ax_continuous):
            for j, x in enumerate(ax):

                ax_code = np.array(
                    [b <= x for _, b in sorted(self.bins.items(), key=lambda t: t[0])] +
                    [x < b for _, b in sorted(self.bins.items(), key=lambda t: t[0])],
                    dtype=bool
                )
                ax_bits[i, j * 2 * self.n_bins: (j + 1) * 2 * self.n_bins] = ax_code

        return csc_matrix(ax_bits)

    def inverse_transform(self, sax_bits, agg='mean'):

        ax_bits = sax_bits.toarray()[0]
        ax_discrete = np.zeros(ax_bits.shape[0], int(ax_bits.shape[1] / (2 * self.n_bins)))

        for i in range(ax_discrete.shape[0]):
            for j in range(ax_discrete.shape[1]):

                # Get bounds
                l_upper_bounds = np.where(ax_bits[((j * 2) + 1) * self.n_bins: (j + 1) * 2 * self.n_bins])[0]
                l_lower_bounds = np.where(ax_bits[j * 2 * self.n_bins: ((j * 2) + 1) * self.n_bins])[0]
                l_bounds = [min([self.bins[k] for k in l_upper_bounds]), max([self.bins[k] for k in l_lower_bounds])]

                if agg == 'mean':
                    ax_discrete[i, j] = np.mean(l_bounds)

                elif agg == 'min':
                    ax_discrete[i, j] = min(l_bounds)

                elif agg == 'max':
                    ax_discrete[i, j] = max(l_bounds)

                else:
                    raise ValueError("choose agg  in {}".format(['mean', 'max', 'min']))

        return ax_discrete

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

