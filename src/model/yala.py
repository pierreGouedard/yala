# Global import
from firing_graph.core.tools.helpers.servers import ArrayServer
from firing_graph.core.solver.sampler import SupervisedSampler
from firing_graph.core.solver.drainer import FiringGraphDrainer
import numpy as np

# Local import
from .utils import build_firing_graph, augment_multi_output_patterns, set_score_params
from .patterns import YalaBasePattern, YalaPredictingPattern


class Yala(object):
    """
    Yala implement a Classifier algorithm based on the optimisation of firing graph (link paper). The classifier support
    multi-class classification and there is no restriction of the cardinality of each class. The Class implement
    the fundamentals of the BaseEstimator, BaseDecisionTree and ClassifierMixin interface from scikit-learn.
    In addition it implements model specific method of interest.

    """
    # TODO:
    #  * P1: Deal with performance issue in various function of src.model.utils.py (slow)
    #  * P2: Enable having p, q for each outputs and to store different precision value for each structure
    #  The above is theoretically working. needs to be tested
    #  * P2: Enable the use of server, how does it affect things ?

    def __init__(self,
                 sampling_rate=0.8,
                 n_sampled_vertices=10,
                 max_iter=5,
                 max_retry=2,
                 learning_rate=5e-2,
                 batch_size=1000,
                 firing_graph=None,
                 min_firing=10,
                 max_precision=0.9,
                 treshold_precision=0.5,
                 overlap_rate=0.5,
                 colinear_penalisation='soft'
                 ):

        # Core parameter of the algorithm
        self.sampling_rate = sampling_rate
        self.n_sampled_vertices = n_sampled_vertices
        self.max_iter = max_iter
        self.max_retry = max_retry
        self.learning_rate = learning_rate
        self.max_precision = max_precision
        self.treshold_precision = treshold_precision
        self.overlap_rate = overlap_rate
        self.batch_size = batch_size
        self.colinear_penalisation = colinear_penalisation

        # Core attributes
        self.firing_graph = firing_graph
        self.drainer_params = {'t': -1}
        self.min_firing = min_firing
        self.n_outputs, self.n_inputs = None, None
        self.server, self.sampler, self.drainer = None, None, None

    def __init_parameters(self, y):
        """

        :param y:
        :return:
        """

        # Set core params from signal and current firing graph)
        ax_precision = np.asarray(y.sum(axis=0) / y.shape[0])[0]

        # Get scoring process params
        ax_p, ax_r = set_score_params(ax_precision, ax_precision * (1 + self.learning_rate))
        ax_weight = ((ax_p - (ax_precision * (ax_p + ax_r))) * self.min_firing).astype(int) + 1
        self.drainer_params.update({'p': ax_p, 'r': ax_r})

        return ax_precision * (1 + self.learning_rate), ax_weight

    def __core_parameters(self, l_patterns, ax_precision):
        """

        :param l_patterns:
        :return:
        """

        # Set current precision for each structure
        ax_precision_ = ax_precision.copy()
        for pattern in l_patterns:
            ax_precision_[pattern.index_output] = min(pattern.precision, ax_precision_[pattern.index_output])
        ax_precision = ax_precision_.clip(ax_precision, self.max_precision / (1 + self.learning_rate))

        # Get corresponding reward / penalty and update drainer_params
        ax_p, ax_r = set_score_params(ax_precision, ax_precision * (1 + self.learning_rate))
        ax_weights = ((ax_p - (ax_precision * (ax_p + ax_r))) * self.min_firing).astype(int) + 1
        self.drainer_params.update({'p': ax_p, 'r': ax_r})

        return ax_precision * (1 + self.learning_rate), ax_weights

    def fit(self, X, y, update=False):
        """

        :param X:
        :param y:
        :param update:
        :return:
        """

        # Initialisation
        if not update:
            self.firing_graph = None

        self.server = ArrayServer(
            X, y, pattern_backward=self.firing_graph, colinear_penalisation=self.colinear_penalisation
        ).stream_features()

        self.n_inputs, self.n_outputs = X.shape[1], y.shape[1]
        self.batch_size = min(y.shape[0], self.batch_size)

        self.sampler = SupervisedSampler(
            self.server, X.shape[1], y.shape[1], self.batch_size, self.sampling_rate, self.n_sampled_vertices,
            firing_graph=self.firing_graph
        )

        # Core loop
        n_no_update = 0
        for i in range(self.max_iter):
            print("[YALA]: Iteration {}".format(i))

            # infer params from signal
            ax_precision, ax_weights = self.__init_parameters(y)

            # Initial sampling
            firing_graph = build_firing_graph(self.sampler.generative_sampling(), ax_weights)
            stop, n, l_patterns = False, 0, []

            # Core loop
            while not stop:

                # Drain firing graph
                firing_graph = FiringGraphDrainer(firing_graph, self.server, self.batch_size, **self.drainer_params)\
                    .drain_all(n_max=y.shape[0])\
                    .firing_graph

                # Augment firing graph with remaining samples
                l_patterns, nu = augment_multi_output_patterns(
                    X, y, firing_graph, self.drainer_params, ax_weights, self.min_firing, self.overlap_rate
                )

                # Compute stop criteria
                stop = (nu == 0)

                if not stop:

                    # update parameters
                    ax_precision, ax_weights = self.__core_parameters(l_patterns, ax_precision)
                    print("[YALA]: {} pattern updated, targeted precision are {}".format(nu, ax_precision))

                    # Sample
                    self.sampler.patterns = l_patterns
                    firing_graph = build_firing_graph(self.sampler.discriminative_sampling(), ax_weights)

                    n += 1

            # Escape main loop on lax retry condition
            if len(l_patterns) == 0:
                n_no_update += 1
                if n_no_update >= self.max_retry:
                    break
            else:
                n_no_update = 0

            # Merge firing graph
            if self.firing_graph is None:
                self.firing_graph = YalaPredictingPattern.from_base_patterns(l_base_patterns=l_patterns)

            else:
                self.firing_graph.augment(l_patterns)

            # Update sampler attributes
            self.server.pattern_backward = self.firing_graph
            self.sampler.patterns = None

        return self

    def predict(self, X):
        """

        :param X:
        :return:
        """
        ax_probas = self.predict_probas(X)
        if ax_probas.shape[1] == 1:
            ax_preds = (ax_probas[:, 0] > self.treshold_precision).astype(int)

        else:
            ax_preds = ax_probas.argmax(axis=1)
        return ax_preds

    def predict_probas(self, X):
        """

        :param X:
        :return:
        """
        assert self.firing_graph is not None, "First fit firing graph"
        ax_probas = np.zeros((X.shape[0], self.firing_graph.O.shape[1]))

        for partition in self.firing_graph.partitions:
            base_pattern = YalaBasePattern.from_partition(partition, self.firing_graph)
            sax_probas = base_pattern.propagate(X).multiply(base_pattern.precision)

            for i, j in zip(*sax_probas.nonzero()):
                if ax_probas[i, j] < sax_probas[i, j]:
                    ax_probas[i, j] = base_pattern.precision

        return ax_probas

    def score(self, X, y):
        raise NotImplementedError
