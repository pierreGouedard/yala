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
    #  * P1: Deal with data types of forward and backward in drainer.
    #  * P1: Deal with format of the matrices of structure (lil_matrix in write mode, csc_matrix in read)
    #  * P1: Enable having p, q for each outputs and to store different precision value for each structure
    #  * P2: Allow custom penalty / reward computation using output imputer + firing_graph already designed
    #  * P2: Enable the use of streamer, how does it affect things ?
    #  * P2: Deal with performance issue in various function of src.model.utils.py (if any)
    #  * P*: Change name of fucking imputer everywhere DONE

    def __init__(self,
                 sampling_rate=0.8,
                 n_sampled_vertices=10,
                 max_iter=5,
                 learning_rate=5e-2,
                 batch_size=1000,
                 firing_graph=None,
                 t=None,
                 min_firing=10,
                 max_precision=0.9,
                 treshold_precision=0.5,
                 overlap_rate=0.5
                 ):

        # Core parameter of the algorithm
        self.sampling_rate = sampling_rate
        self.n_sampled_vertices = n_sampled_vertices
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_precision = max_precision
        self.treshold_precision = treshold_precision
        self.overlap_rate = overlap_rate
        self.batch_size = batch_size

        # Core attributes
        self.firing_graph = firing_graph
        self.drainer_params = {'t': -1, 'min_firing': min_firing, 'batch_size': batch_size}
        self.min_firing = min_firing
        self.n_outputs, self.n_inputs = None, None
        self.server, self.sampler, self.drainer = None, None, None

    def __init_parameters(self, y):
        """

        :param y:
        :return:
        """

        # Set batch sizefrom signal
        n_max = y.shape[0]

        # Set core params from signal and current firing graph
        init_precision = np.array(y.sum(axis=0) / y.shape[0]).min()
        precision = init_precision * (1 + self.learning_rate)

        # Get scoring process params
        p, q = set_score_params(init_precision, precision)

        self.drainer_params.update({
            'n_max': n_max, 'p': p, 'q': q, 'weight': (p - init_precision * (p + q)) * self.min_firing,
            'precision': init_precision
        })

    def __core_parameters(self, l_patterns):
        """

        :param l_patterns:
        :return:
        """

        # Set core params from signal and current firing graph
        min_precision = min([structure.precision for structure in l_patterns if structure.precision is not None])
        init_precision = max(min_precision, self.drainer_params['precision'])
        precision = init_precision * (1 + self.learning_rate)

        if precision > self.max_precision:
            init_precision, precision = self.max_precision - 5e-2, self.max_precision

        p, q = set_score_params(init_precision, precision)

        self.drainer_params.update({
            'p': p, 'q': q, 'weight': int((p - init_precision * (p + q)) * self.min_firing) + 1, 'precision': precision
        })

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

        self.server = ArrayServer(X, y)
        self.server.stream_features()

        self.n_inputs, self.n_outputs = X.shape[1], y.shape[1]
        self.batch_size = min(y.shape[0], self.batch_size)

        self.sampler = SupervisedSampler(
            self.server, X.shape[1], y.shape[1], self.batch_size, self.sampling_rate, self.n_sampled_vertices,
            firing_graph=self.firing_graph
        )

        # Core loop
        for i in range(self.max_iter):
            print("[YALA]: Iteration {}".format(i))

            # infer params from signal
            self.__init_parameters(y)

            # Initial sampling
            firing_graph = build_firing_graph(self.sampler.generative_sampling(), self.drainer_params)
            stop, n, l_patterns = False, 0, []

            # Core loop
            while not stop:

                # Drain firing graph
                firing_graph = FiringGraphDrainer(firing_graph, self.server, self.batch_size)\
                    .drain_all(n_max=self.drainer_params['n_max'])\
                    .firing_graph

                # Augment firing graph with remaining samples
                l_patterns, nu = augment_multi_output_patterns(
                    self.overlap_rate, self.drainer_params, self.min_firing, firing_graph, X, y
                )

                # Compute stop criteria
                stop = (nu == 0)

                if not stop:

                    # update parameters
                    self.__core_parameters(l_patterns)

                    print("[YALA]: {} pattern updated, targeted precision is {}".format(
                        nu, self.drainer_params['precision']
                    ))

                    # Sample
                    self.sampler.patterns = l_patterns
                    firing_graph = build_firing_graph(self.sampler.discriminative_sampling(), self.drainer_params)

                    n += 1

            # Merge firing graph
            if self.firing_graph is None:
                self.firing_graph = YalaPredictingPattern.from_base_patterns(l_base_patterns=l_patterns)

            else:
                self.firing_graph.augment(l_patterns)

            # Update sampler attributes
            self.sampler.firing_graph = self.firing_graph
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
