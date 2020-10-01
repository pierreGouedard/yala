# Global import
from firing_graph.core.tools.helpers.servers import ArrayServer
from firing_graph.core.tools.helpers.sampler import YalaSampler
from firing_graph.core.solver.drainer import FiringGraphDrainer
import numpy as np
from scipy.sparse import lil_matrix, csc_matrix

# Local import
from .utils import build_firing_graph, disclose_patterns_multi_output, set_feedbacks
from .patterns import YalaBasePatterns


class Yala(object):
    """
    Yala implement a Classifier algorithm based on the optimisation of firing graph (link paper). The classifier support
    multi-class classification and there is no restriction of the cardinality of each class. The Class implement
    the fundamentals of the BaseEstimator, BaseDecisionTree and ClassifierMixin interface from scikit-learn.
    In addition it implements model specific method of interest.

    """

    def __init__(self,
                 sampling_rate=0.8,
                 max_iter=5,
                 max_retry=5,
                 min_gain=1e-3,
                 margin=2e-2,
                 drainer_bs=500,
                 selection_bs=None,
                 sampler_bs=None,
                 batch_size=1000,
                 min_firing=10,
                 min_precision=0.75,
                 max_precision=None,
                 n_overlap=100,
                 init_eval_score=0,
                 dropout_mask=0.2,
                 max_candidate=100
                 ):

        # Core parameter of the algorithm
        self.sampling_rate = sampling_rate
        self.max_iter = max_iter
        self.max_retry = max_retry
        self.min_gain = min_gain
        self.margin = margin
        self.min_precision = min_precision
        self.eval_score = init_eval_score
        self.dropout_mask = dropout_mask
        self.max_candidate = max_candidate

        if max_precision is None:
            self.max_precision = 1 - (2 * self.min_gain)
        else:
            self.max_precision = max_precision

        self.drainer_bs = drainer_bs
        if selection_bs is None:
            self.selection_bs = drainer_bs
        else:
            self.selection_bs = selection_bs

        if sampler_bs is None:
            self.sampler_bs = drainer_bs
        else:
            self.sampler_bs = sampler_bs

        self.batch_size = batch_size

        # Core attributes
        self.firing_graph = None
        self.drainer_params = {'t': -1}
        self.min_firing = min_firing
        self.n_overlap = n_overlap
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
        ax_p, ax_r = set_feedbacks(ax_precision + self.min_gain, ax_precision + (2 * self.min_gain))
        ax_weight = ((ax_p - (ax_precision * (ax_p + ax_r))) * self.min_firing).astype(int) + 1
        self.drainer_params.update({'p': ax_p, 'r': ax_r})

        return ax_precision + self.min_gain, ax_weight

    def __core_parameters(self, l_patterns):
        """

        :param l_patterns:
        :return:
        """
        # Set current precision for each structure
        ax_precision = np.array([p.precision for p in sorted(l_patterns, key=lambda x: x.output_id)])
        ax_lower_precision = (ax_precision - (2 * self.margin)).clip(min=self.min_gain)
        ax_upper_precision = (ax_precision - self.margin).clip(min=2 * self.min_gain)

        # Get corresponding reward / penalty and update drainer_params
        ax_p, ax_r = set_feedbacks(ax_lower_precision, ax_upper_precision)
        ax_weights = ((ax_p - (ax_lower_precision * (ax_p + ax_r))) * self.min_firing).astype(int) + 1
        self.drainer_params.update({'p': ax_p, 'r': ax_r})

        return ax_upper_precision.round(3), ax_weights

    def fit(self, X, y, eval_set=None, scoring=None, sample_weight=None, mapping_feature_input=None):
        """row

        :param X:
        :param y:
        :param eval_set:
        :param sample_weight:
        :return:
        """
        self.server = ArrayServer(X, y, dropout_mask=self.dropout_mask).stream_features()
        self.n_inputs = X.shape[1]
        # TODO: Create encoder
        # TODO: forget about defining batch size different for sampler, drainer end selection, there is onl one batch_
        #  size and only 1 max_draining_iteration
        self.batch_size = min(y.shape[0], self.batch_size)
        self.drainer_bs, self.selection_bs = min(y.shape[0], self.drainer_bs), min(y.shape[0], self.selection_bs)
        self.sampler_bs = min(y.shape[0], self.drainer_bs)

        # New sampler
        self.sampler = YalaSampler(mapping_feature_input, self.sampling_rate)

        # Core loop
        import time
        start = time.time()
        count_no_update, l_dropouts = 0, []
        for i in range(self.max_iter):
            print("[YALA]: Iteration {}".format(i))

            # infer params from signal
            ax_precision, ax_weights = self.__init_parameters(y)

            # Initial sampling
            firing_graph = build_firing_graph(
                self.sampler.sample(self.server.n_label), ax_weights, n_inputs=self.n_inputs
            )
            stop, n, l_selected = False, 0, []

            # Core loop
            while not stop:

                # Drain firing graph
                firing_graph = FiringGraphDrainer(
                    firing_graph, self.server, self.drainer_bs, **self.drainer_params
                )\
                    .drain_all(n_max=self.batch_size)\
                    .firing_graph

                # Disclose new patterns
                # TODO: cannot really test this guy before refactoring sampler properly
                # TODO: step 1: remove any reference to pred patterns
                #       step 3: test disclose pattern
                l_transients, l_selected = disclose_patterns_multi_output(
                    l_selected, self.server, self.selection_bs, firing_graph, self.drainer_params, ax_weights,
                    self.min_firing, self.n_overlap, self.min_precision, self.max_precision, self.min_gain,
                    self.max_candidate, csc_matrix(mapping_feature_input)
                )

                print("[YALA]: {} pattern disclosed".format(len(l_transients)))
                stop = (len(l_transients) == 0)

                if not stop:

                    # update parameters
                    ax_precision, ax_weights = self.__core_parameters(l_transients)

                    # Sample
                    firing_graph = build_firing_graph(self.sampler.sample(len(l_transients)), ax_weights, l_transients)
                    n += 1

                    print("[YALA]: {} pattern updated, targeted precision are {}".format(
                        len(l_transients), ax_precision)
                    )

            if self.firing_graph is not None:
                self.firing_graph = self.firing_graph.augment(
                    l_selected, max([p['group_id'] for p in self.firing_graph.partitions]) + 1
                )

            else:
                self.firing_graph = YalaBasePatterns.from_patterns(l_selected, group_id=0)

            # Escape main loop on last retry condition
            if not len(l_selected) > 0:
                count_no_update += 1
                if count_no_update > self.max_retry:
                    break
            else:
                count_no_update = 0

            # Update sampler attributes
            self.server.pattern_mask = self.firing_graph.copy()
            self.server.sax_mask_forward = None
            self.server.pattern_backward = None

        print('duration of algorithm: {}s'.format(time.time() - start))
        return self

    def predict_proba(self, X, n_label, min_probas=0.5):
        """

        :param X:
        :return:
        """
        # TODO: Encode features here
        assert self.firing_graph is not None, "First fit firing graph"

        l_partitions = [p for p in sorted(self.firing_graph.partitions, key=lambda x: x['indices'][0])]

        ax_probas = self.firing_graph\
            .group_output()\
            .propagate_value(X, np.array([p['precision'] for p in l_partitions])).A

        sax_sum = lil_matrix((ax_probas.shape[1], n_label), dtype=bool)
        for label in range(n_label):
            l_indices = [p['group_id'] for p in l_partitions if p['label_id'] == label]
            sax_sum[l_indices, label] = True

        ax_count = sax_sum.sum(axis=0).A[0]
        ax_probas = ax_probas.clip(min=min_probas).dot(sax_sum.A)

        # Merge labels
        ax_probas = ax_probas.dot(np.eye(n_label) * 2 - np.ones((n_label, n_label)))
        ax_probas += (np.eye(n_label) * -1 + np.ones((n_label, n_label))).dot(ax_count)
        ax_probas /= ax_count.sum()

        self.firing_graph.ungroup_output()

        return ax_probas
