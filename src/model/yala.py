# Global import
from firing_graph.tools.helpers.servers import ArrayServer
from firing_graph.tools.helpers.sampler import YalaSampler
from firing_graph.solver.drainer import FiringGraphDrainer
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
                 draining_size=500,
                 batch_size=1000,
                 min_firing=10,
                 min_precision=0.75,
                 max_precision=None,
                 n_overlap=100,
                 init_eval_score=0,
                 dropout_rate_mask=0.2,
                 ):

        # Core parameter of the algorithm
        self.sampling_rate = sampling_rate
        self.max_iter = max_iter
        self.max_retry = max_retry
        self.min_gain = min_gain
        self.margin = margin
        self.min_precision = min_precision
        self.eval_score = init_eval_score
        self.dropout_rate_mask = dropout_rate_mask

        if max_precision is None:
            self.max_precision = 1 - (2 * self.min_gain)
        else:
            self.max_precision = max_precision

        self.draining_size = draining_size
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
        """
        row

        :param X:
        :param y:
        :param eval_set:
        :param sample_weight:
        :return:
        """
        self.server = ArrayServer(X, y, dropout_rate_mask=self.dropout_rate_mask).stream_features()
        self.n_inputs = X.shape[1]
        # TODO: Create encoder
        # TODO: forget about defining batch size different for sampler, drainer end selection, there is onl one batch_
        #  size and only 1 max_draining_iteration
        self.batch_size = min(y.shape[0], self.batch_size)
        self.draining_size = min(y.shape[0], self.draining_size)

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
            stop, n, l_selected, n_debug = False, 0, [], 0

            # Core loop
            while not stop:
                # Drain firing graph
                firing_graph = FiringGraphDrainer(
                    firing_graph, self.server, self.batch_size, **self.drainer_params
                )\
                    .drain_all(n_max=self.draining_size)\
                    .firing_graph

                # Disclose new patterns
                l_transients, l_selected = disclose_patterns_multi_output(
                    l_selected, self.server, self.batch_size, firing_graph, self.drainer_params, ax_weights,
                    self.min_firing, self.n_overlap, self.min_precision, self.max_precision, self.min_gain,
                    csc_matrix(mapping_feature_input)
                )

                print("[YALA]: {} pattern disclosed".format(len(l_transients)))
                stop = (len(l_transients) == 0)

                if not stop:

                    # update parameters
                    ax_precision, ax_weights = self.__core_parameters(l_transients)

                    # Sample
                    firing_graph = build_firing_graph(
                        self.sampler.sample(len(l_transients), l_transients), ax_weights, l_transients
                    )
                    n += 1

                    print("[YALA]: {} pattern updated, targeted precision are {}".format(
                        len(l_transients), ax_precision)
                    )

            if self.firing_graph is not None:
                self.firing_graph = self.firing_graph.augment_from_patterns(
                    l_selected, 'same', **{'group_id': max([p['group_id'] for p in self.firing_graph.partitions]) + 1}
                )

            else:
                self.firing_graph = YalaBasePatterns.from_patterns(l_selected, 'same', **{'group_id': 0})

            # Escape main loop on last retry condition
            if not len(l_selected) > 0:
                count_no_update += 1
                if count_no_update > self.max_retry:
                    break
            else:
                count_no_update = 0

            # Update sampler attributes
            self.server.update_mask(YalaBasePatterns.from_patterns(l_selected, output_method='same'))
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

        # group_id in partition
        sax_probas = self.firing_graph\
            .reset_output(l_outputs=[p['group_id'] * 2 + p['label_id'] for p in self.firing_graph.partitions])\
            .propagate_value(X, np.array([p['precision'] for p in l_partitions]))

        sax_sum = lil_matrix((sax_probas.shape[1], n_label), dtype=bool)
        for label in range(n_label):
            l_indices = [p['group_id'] * 2 + p['label_id'] for p in l_partitions if p['label_id'] == label]
            sax_sum[l_indices, label] = True

        ax_count = sax_sum.sum(axis=0).A
        ax_probas = sax_probas.dot(sax_sum).A
        ax_probas += (ax_count.repeat(sax_probas.shape[0], axis=0) - (sax_probas > 0).dot(sax_sum).A) * 0.5

        # Merge labels
        ax_probas = ax_probas.dot(np.eye(n_label) * 2 - np.ones((n_label, n_label)))
        ax_probas += (np.eye(n_label) * -1 + np.ones((n_label, n_label))).dot(ax_count[0])
        ax_probas /= ax_count.sum()

        self.firing_graph.reset_output(key='label_id')

        return ax_probas
