# Global import
from firing_graph.tools.helpers.servers import ArrayServer
from firing_graph.tools.helpers.sampler import YalaSampler
from firing_graph.solver.drainer import FiringGraphDrainer
import numpy as np
from scipy.sparse import lil_matrix
from dataclasses import asdict

# Local import
from .utils import build_draining_firing_graph, set_feedbacks
from .patterns import YalaBasePatterns
from .picker import YalaGreedyPicker, YalaOrthogonalPicker
from .data_models import DrainerFeedbacks, DrainerParameters


class Yala(object):
    """
    Yala implement a Classifier algorithm based on the optimisation of firing graph (link paper). The classifier support
    multi-class classification and there is no restriction of the cardinality of each class. The Class implement
    the fundamentals of the BaseEstimator, BaseDecisionTree and ClassifierMixin interface from scikit-learn.
    In addition it implements model specific method of interest.

    """

    def __init__(self,
                 max_iter=5,
                 max_retry=5,
                 min_gain=1e-3,
                 margin=2e-2,
                 draining_size=500,
                 batch_size=1000,
                 sampling_rate=0.8,
                 dropout_rate_mask=0.2,
                 picker_type='orthogonal',
                 min_firing=10,
                 min_precision=0.75,
                 max_precision=None,
                 n_overlap=100,
                 k=2,
                 ):

        # Sampler params
        self.sampling_rate = sampling_rate

        # Core paramerters
        self.max_iter, self.max_retry = max_iter, max_retry
        self.min_gain, self.margin, self.min_firing = min_gain, margin, min_firing

        # Server params
        self.dropout_rate_mask = dropout_rate_mask

        # Picker params
        self.picker_type = picker_type
        self.picker_params = dict(
            min_gain=min_gain, min_precision=min_precision, min_firing=min_firing,
            max_precision=1 - (2 * self.min_gain) if max_precision is None else max_precision,
        )
        self.n_overlap, self.k = n_overlap, k

        # Drainer params
        self.drainer_params = DrainerParameters(feedbacks=None, weights=None)
        self.draining_size = draining_size
        self.batch_size = batch_size

        # Yala Core attributes
        self.firing_graph = None
        self.server, self.sampler, self.drainer, self.picker = None, None, None, None

    def __init_parameters(self, y):
        """

        :param y:
        :return:
        """
        # Set core params from signal and current firing graph)
        ax_precision = np.asarray(y.sum(axis=0) / y.shape[0])[0]

        # Get scoring process params
        ax_p, ax_r = set_feedbacks(ax_precision + self.min_gain, ax_precision + (2 * self.min_gain))
        drainer_params = DrainerParameters(
            feedbacks=DrainerFeedbacks(penalties=ax_p, rewards=ax_r),
            weights=((ax_p - (ax_precision * (ax_p + ax_r))) * self.min_firing).astype(int) + 1
        )
        return drainer_params

    def fit(self, X, y, eval_set=None, scoring=None, sample_weight=None, mapping_feature_input=None):
        """
        row

        :param X:
        :param y:
        :param eval_set:
        :param sample_weight:
        :return:
        """
        # TODO: Server should implement encoder and create a big init function

        # Update size parameters if necessary
        self.batch_size = min(y.shape[0], self.batch_size)
        self.draining_size = min(y.shape[0], self.draining_size)

        # Instantiate core components
        self.server = ArrayServer(X, y, dropout_rate_mask=self.dropout_rate_mask).stream_features()

        self.sampler = YalaSampler(mapping_feature_input, self.server.n_label, self.sampling_rate)
        if self.picker_type == 'greedy':
            self.picker = YalaGreedyPicker(
                self.server.next_forward(self.batch_size, update_step=False).sax_data_forward, self.n_overlap,
                **dict(n_label=self.server.n_label, mapping_feature_input=mapping_feature_input, **self.picker_params)
            )
        elif self.picker_type == 'orthogonal':
            self.picker = YalaOrthogonalPicker(
                self.k,
                **dict(n_label=self.server.n_label, mapping_feature_input=mapping_feature_input, **self.picker_params)
            )

        # Core loop
        import time
        start = time.time()
        count_no_update, l_dropouts = 0, []
        for i in range(self.max_iter):
            print("[YALA]: Iteration {}".format(i))

            # infer init params from signal and build initial graph
            self.drainer_params = self.__init_parameters(y)
            firing_graph = build_draining_firing_graph(self.sampler, self.drainer_params)

            stop = False
            while not stop:
                # Drain firing graph
                firing_graph = FiringGraphDrainer(
                    firing_graph, self.server, self.batch_size, **asdict(self.drainer_params.feedbacks)
                )\
                    .drain_all(n_max=self.draining_size)\
                    .firing_graph

                # Pick partial patterns
                partials, self.drainer_params = self.picker.pick_patterns_multi_label(self.server, firing_graph)

                # Update stop criteria
                stop = (len(partials.partitions) == 0)
                if not stop:
                    firing_graph = build_draining_firing_graph(self.sampler, self.drainer_params, partials)

            # Augment current firing graph
            if self.firing_graph is not None:
                self.firing_graph = self.firing_graph.augment_from_pattern(
                    self.picker.completes, 'same', **{'group_id': i}
                )

            else:
                # TODO: what about 'group_id'
                self.firing_graph = self.picker.completes.copy() if self.picker.completes is not None else None

            # Escape main loop on last retry condition
            if self.picker.completes is None:
                count_no_update += 1
                if count_no_update > self.max_retry:
                    break
            else:
                count_no_update = 0

            # Update sampler attributes
            self.server.update_mask(self.picker.completes)
            self.picker.completes = None
            self.server.sax_mask_forward = None
            self.server.pattern_backward = None

        print('duration of algorithm: {}s'.format(time.time() - start))
        return self

    def predict_proba_new(self, X, n_label):
        """

        :param X:
        :return:
        """
        assert self.firing_graph is not None, "First fit firing graph"
        l_partitions = [p for p in sorted(self.firing_graph.partitions, key=lambda x: x['indices'][0])]

        # group_id in partition
        ax_probas = self.firing_graph\
            .propagate_value(X, np.array([p['precision'] for p in l_partitions])).A

        # Merge labels
        ax_probas = ax_probas.dot(np.eye(n_label) * 2 - np.ones((n_label, n_label)))
        ax_probas = (ax_probas + 1) / 2

        return ax_probas

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
        ax_probas += (ax_count.repeat(sax_probas.shape[0], axis=0) - (sax_probas > 0).dot(sax_sum).A) * min_probas

        # Merge labels
        ax_probas = ax_probas.dot(np.eye(n_label) * 2 - np.ones((n_label, n_label)))
        ax_probas += (np.eye(n_label) * -1 + np.ones((n_label, n_label))).dot(ax_count[0])
        ax_probas /= ax_count.sum()

        self.firing_graph.reset_output(key='label_id')

        return ax_probas
