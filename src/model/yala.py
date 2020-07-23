# Global import
from firing_graph.core.tools.helpers.servers import ArrayServer
from firing_graph.core.solver.sampler import SupervisedSampler
from firing_graph.core.solver.drainer import FiringGraphDrainer
import numpy as np
from scipy.sparse import lil_matrix
import time

# Local import
from .utils import build_firing_graph, disclose_patterns_multi_output, set_feedbacks
from .patterns import YalaPredPatterns


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
                 drainer_batch_size=500,
                 batch_size=1000,
                 min_firing=10,
                 min_precision=0.75,
                 max_precision=None,
                 overlap_rate=0.5,
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

        self.drainer_batch_size = drainer_batch_size
        self.batch_size = batch_size

        # Core attributes
        self.firing_graph = None
        self.drainer_params = {'t': -1}
        self.min_firing = min_firing
        self.overlap = int(self.min_firing * (self.drainer_batch_size / self.batch_size) * overlap_rate)
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

    def fit(self, X, y, eval_set=None, scoring=None, sample_weight=None):
        """row

        :param X:
        :param y:
        :param eval_set:
        :param sample_weight:
        :return:
        """

        self.server = ArrayServer(X, y, dropout_mask=self.dropout_mask).stream_features()
        self.n_inputs, self.n_outputs = X.shape[1], y.shape[1]
        self.batch_size = min(y.shape[0], self.batch_size)
        self.drainer_batch_size = min(y.shape[0], self.drainer_batch_size)
        self.sampler = SupervisedSampler(self.server, self.drainer_batch_size, self.sampling_rate)

        # Core loop
        count_no_update, l_dropouts = 0, []
        for i in range(self.max_iter):
            print("[YALA]: Iteration {}".format(i))

            # infer params from signal
            ax_precision, ax_weights = self.__init_parameters(y)

            # Initial sampling
            firing_graph = build_firing_graph(
                self.sampler.generative_sampling(), ax_weights, n_inputs=self.n_inputs, n_outputs=self.n_outputs
            )
            stop, n, l_selected = False, 0, []

            # Core loop
            while not stop:

                # Drain firing graph
                firing_graph = FiringGraphDrainer(
                    firing_graph, self.server, self.drainer_batch_size, **self.drainer_params
                )\
                    .drain_all(n_max=self.batch_size)\
                    .firing_graph

                # Disclose new patterns
                self.sampler.patterns, l_selected = disclose_patterns_multi_output(
                    l_selected, self.server, self.drainer_batch_size, firing_graph, self.drainer_params, ax_weights,
                    self.min_firing, self.overlap, self.min_precision, self.max_precision, self.min_gain,
                    self.max_candidate
                )

                print("[YALA]: {} pattern disclosed".format(len(self.sampler.patterns)))
                stop = (len(self.sampler.patterns) == 0)

                if not stop:

                    # update parameters
                    ax_precision, ax_weights = self.__core_parameters(self.sampler.patterns)

                    # Sample
                    firing_graph = build_firing_graph(self.sampler.discriminative_sampling(), ax_weights)
                    n += 1

                    print("[YALA]: {} pattern updated, targeted precision are {}".format(
                        len(self.sampler.patterns), ax_precision)
                    )

            if self.firing_graph is not None:
                self.firing_graph = self.firing_graph.augment(
                    l_selected, max([p['group_id'] for p in self.firing_graph.partitions]) + 1
                )

            else:
                self.firing_graph = YalaPredPatterns.from_pred_patterns(l_selected, group_id=0)

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
            self.sampler.patterns = None

        return self

    def predict(self, X):
        """

        :param X:
        :return:
        """
        ax_preds = self.firing_graph.propagate(X).A.astype(int)
        if ax_preds.shape[1] == 1:
            ax_preds = ax_preds[:, 0]

        else:
            raise NotImplementedError

        return ax_preds

    def predict_proba(self, X, n_label, min_probas=0.5):
        """

        :param X:
        :return:
        """
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

    def predict_score(self, X, n_label, min_score=0):
        """

        :param X:
        :return:
        """
        assert self.firing_graph is not None, "First fit firing graph"

        l_partitions = [p for p in sorted(self.firing_graph.partitions, key=lambda x: x['indices'][0])]

        ax_scores = self.firing_graph\
            .group_output()\
            .propagate_value(X, [p['score'] for p in l_partitions]).A

        sax_sum = lil_matrix((ax_scores.shape[1], n_label), dtype=bool)
        for label in range(n_label):
            l_indices = [p['group_id'] for p in l_partitions if p['label_id'] == label]
            sax_sum[l_indices, label] = True

        ax_count = sax_sum.sum(axis=0).A[0]
        ax_scores = ax_scores.clip(min=min_score).dot(sax_sum.A)

        # Merge labels
        ax_scores = ax_scores.dot(np.eye(n_label) * 2 - np.ones((n_label, n_label)))
        ax_scores += (np.eye(n_label) * -1 + np.ones((n_label, n_label))).dot(ax_count)
        ax_scores /= ax_count.sum()

        self.firing_graph.ungroup_output()

        return ax_scores
