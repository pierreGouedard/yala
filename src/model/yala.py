# Global import
from firing_graph.solver.drainer import FiringGraphDrainer
import numpy as np
from scipy.sparse import lil_matrix, csc_matrix
from dataclasses import asdict

# Local import
from .utils import build_draining_firing_graph, set_feedbacks
from src.model.helpers.picker import YalaGreedyPicker, YalaOrthogonalPicker
from src.model.helpers.sampler import YalaSampler
from src.model.helpers.server import YalaUnclassifiedServer, YalaMisclassifiedServer
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
                 picker_type='greedy',
                 min_firing=10,
                 min_precision=0.75,
                 max_precision=None,
                 n_overlap=100,
                 server_type='misclassified'
                 ):

        # Sampler params
        self.sampling_rate = sampling_rate

        # Core parameters
        self.max_iter, self.max_retry = max_iter, max_retry

        # Server params
        self.dropout_rate_mask = dropout_rate_mask

        # Picker params
        self.picker_type = picker_type
        self.picker_params = dict(
            min_gain=min_gain, min_precision=min_precision, min_firing=min_firing, margin=margin,
            max_precision=1 - (2 * min_gain) if max_precision is None else max_precision,
        )
        self.n_overlap = n_overlap

        # Drainer params
        self.drainer_params = DrainerParameters(feedbacks=None, weights=None)
        self.draining_size = draining_size
        self.batch_size = batch_size

        # Server params
        self.server_type = server_type

        # Yala Core attributes
        self.firing_graph = None
        self.server, self.sampler, self.drainer, self.picker = None, None, None, None

        # TODO: test / temp
        self.d_merger = {}

    def __init_parameters(self, ax_precision):
        """

        :param ax_precision:
        :return:
        """
        # Get scoring process params
        ax_p, ax_r = set_feedbacks(ax_precision + self.picker.min_gain, ax_precision + (2 * self.picker.min_gain))
        drainer_params = DrainerParameters(
            feedbacks=DrainerFeedbacks(penalties=ax_p, rewards=ax_r),
            weights=((ax_p - (ax_precision * (ax_p + ax_r))) * self.picker.min_firing).astype(int) + 1
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
        # Instantiate core components
        if self.server_type == 'unclassified':
            self.server = YalaUnclassifiedServer(
                1, 0, X, y, **dict(dropout_rate_mask=self.dropout_rate_mask)
            ).stream_features()
        elif self.server_type == 'misclassified':
            self.server = YalaMisclassifiedServer(
                1, 0, X, y, **dict(dropout_rate_mask=self.dropout_rate_mask)
            ).stream_features()
        else:
            raise NotImplementedError

        # Get sampler
        self.sampler = YalaSampler(mapping_feature_input, self.server.n_label, self.sampling_rate)

        if self.picker_type == 'greedy':
            self.picker = YalaGreedyPicker(
                #X[y.A[:, 0], :][:self.batch_size, :]
                self.server.next_forward(self.batch_size, update_step=False).sax_data_forward, self.n_overlap,
                **dict(n_label=self.server.n_label, mapping_feature_input=mapping_feature_input, **self.picker_params)
            )
        elif self.picker_type == 'orthogonal':
            self.picker = YalaOrthogonalPicker(
                **dict(n_label=self.server.n_label, mapping_feature_input=mapping_feature_input, **self.picker_params)
            )

        # TODO: new pardigm
        #   amplify -> drain -> amplify -> drain -> amplify -> drain ->|
        #                                                              |-> stop

        # TODO: test amplifier
        from src.model.patterns import YalaBasePatterns
        from src.model.util_new_paradigm import get_amplifier_firing_graph, get_drainer_firing_graph, \
            split_drained_graph, final_bit_selection, amplify_bits, create_random_fg, get_binomial_upper_ci
        from scipy.ndimage.interpolation import shift
        from matplotlib import pyplot as plt
        import time
        print('============== sample 1 ================')
        plot_path = 'DATA/test_new_paradigm/{}'
        ax_base_activations = X.sum(axis=0).A[0]
        sample = X[0, :]

        self.server.stream_features()
        stop, l_inputs, l_levels, n_it = False, [sample.T], [5], 0
        while not stop:
            print(f"Iteration : {n_it}")

            ########## Amplify
            l_amplifiers = [get_amplifier_firing_graph(sax_i, l_levels[i]) for i, sax_i in enumerate(l_inputs)]
            l_activations = [amp.propagate(X) for amp in l_amplifiers]

            # Print useful information
            print("Amplifier info: \n")
            print(f"activation left: {l_activations[0].sum(axis=0)}, activations right ?")
            print(
                f'threshold selected: '
                f'{round(l_levels[0] / l_amplifiers[0].I[:, 0].astype(bool).T.dot(mapping_feature_input).sum(), 2)}'
            )

            if l_activations[0].tocsc()[:, 0].sum() < 100:

                # TODO: what about the right guy ? to deal with later
                sax_I, level = amplify_bits(
                    current_fg.propagate(X).astype(int).T.dot(X), current_fg.I.A[:, 0], ax_base_activations,
                    current_fg.levels[0], mapping_feature_input, 0.5,
                )

                # Stop Criterion: nothing added (seems the best) <= this one is used for the moment
                ax_feature_new = sax_I.astype(bool).T.dot(mapping_feature_input).A[0]
                ax_feature_current = current_fg.I[:, 0].T.dot(mapping_feature_input).A[0]

                if ax_feature_new.astype(int).dot(ax_feature_current) == ax_feature_new.sum():
                    final_level = sax_I.astype(bool).T.dot(mapping_feature_input).sum() - 1
                    current_fg = get_amplifier_firing_graph(sax_I, final_level)
                    break

            else:

                sax_inner = l_activations[0].astype(int).T.dot(X)
                sax_I, level = amplify_bits(
                    sax_inner, l_amplifiers[0].I.A[:, 0], ax_base_activations, l_levels[0], mapping_feature_input, 0.5,
                )

                if n_it == 0:
                    level = 6

            final_level = sax_I.astype(bool).T.dot(mapping_feature_input).sum() - 1
            current_fg = get_amplifier_firing_graph(sax_I, final_level)

            # Print useful information
            print(f"# feature left {sax_I.astype(bool).T.dot(mapping_feature_input).sum()}, level left: {level}")

            # This could be computed using amplifier
            drainer_fg = get_drainer_firing_graph(sax_I, level)
            sax_activations = drainer_fg.propagate(X)
            target_prec = sax_activations.T.astype(int).dot(y).sum() / sax_activations.sum()

            ########## Drain
            drainer_params = self.__init_parameters(np.array([target_prec]))
            drainer_fg.matrices['Iw'] = sax_I * drainer_params.weights[0]

            print("Drainer info: \n")
            print(f'drainer params: {drainer_params}')
            print(f'activation left: {sax_activations.sum()}, precision left: {target_prec}')

            # split it using drained
            drained = FiringGraphDrainer(
                drainer_fg, self.server, self.batch_size, **asdict(drainer_params.feedbacks)
            ) \
                .drain_all(n_max=self.draining_size) \
                .firing_graph

            sax_I_left, sax_I_right = split_drained_graph(
                drained.Iw, drained.backward_firing['i'], drainer_params.feedbacks.penalties,
                drainer_params.feedbacks.rewards, drainer_params.weights, mapping_feature_input
            )
            print(f"# feature left {sax_I_left.astype(bool).T.dot(mapping_feature_input).sum()}, level: {drained.levels}")

            l_inputs = [sax_I_left]
            l_levels = [level]
            n_it += 1

        # FINAL feature and but selection (diff noise level for denoising)
        final_fg = final_bit_selection(current_fg, mapping_feature_input, X, ax_base_activations, noise_level=2)


        # TODO: the analysis is done may be, before starting to code things clearly, try  to test a simple method to
        #  avoid early failure of draining by a clever selection of which feature to drain 
        # Analysis draining
        drainer_fg = get_drainer_firing_graph(current_fg.matrices['Iw'][:, 0], current_fg.levels[0])
        drainer_params = self.__init_parameters(np.array([0.001]))
        drainer_fg.matrices['Iw'] = sax_I * drainer_params.weights[0]

        # split it using drained
        drained = FiringGraphDrainer(
            drainer_fg, self.server, self.batch_size, **asdict(drainer_params.feedbacks)
        ) \
            .drain_all(n_max=self.draining_size) \
            .firing_graph

        sax_I_left, _ = split_drained_graph(
            drained.Iw, drained.backward_firing['i'], drainer_params.feedbacks.penalties,
            drainer_params.feedbacks.rewards, drainer_params.weights, mapping_feature_input,
            debug=True, save=True
        )

        # Analysis activation
        sax_x_final = final_fg.propagate(X).tocsc()
        prec = sax_x_final[:, 0].T.astype(int).dot(y).A / sax_x_final[:, 0].sum()
        print(f"Final fg at level {final_fg.levels[0]} is has prec {prec} and activate {sax_x_final[:, 0].sum()} times")

        # Analysis compare against lucky sampling
        n_features = final_fg.I[:, 0].T.dot(mapping_feature_input).sum()
        ax_activations = np.zeros((2, n_features))
        for i in range(n_features - 5):
            # Random firing graph
            test_fg = create_random_fg(final_fg, mapping_feature_input, n_features - i)
            ax_activations[0, i] = test_fg.propagate(X).sum()

            final_fg.levels[0] = n_features - i
            ax_activations[1, i] = final_fg.propagate(X).tocsc()[:, 0].sum()

        plt.plot(ax_activations[0, :], color='k')
        plt.plot(ax_activations[1, :], color='b')
        plt.show()

        # TODO:
        #   * implement unefficient method and compare all children
        #   * implement a fucking generic program of the new paradigm that replace entirely the previous way of doing.
        #

        # END

        # Core loop
        import time
        start = time.time()
        count_no_update, l_dropouts = 0, []
        for i in range(self.max_iter):
            print("[YALA]: Iteration {}".format(i))

            # infer init params from signal and build initial graph
            print(self.server.get_init_precision())
            self.drainer_params = self.__init_parameters(self.server.get_init_precision())
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
                stop = partials is None
                if not stop:
                    firing_graph = build_draining_firing_graph(self.sampler, self.drainer_params, partials)

            # Augment current firing graph
            if self.firing_graph is not None:
                self.firing_graph = self.firing_graph.augment_from_pattern(
                    self.picker.completes, 'same', **{'group_id': i}
                )

            else:
                self.firing_graph = self.picker.completes.copy() if self.picker.completes is not None else None

            # Escape main loop on last retry condition
            if self.picker.completes is None:
                count_no_update += 1
                if count_no_update > self.max_retry:
                    break
            else:
                count_no_update = 0

            # Update sampler attributes
            self.server.update_mask_with_pattern(self.picker.completes)
            self.picker.completes = None
            self.server.sax_mask_forward = None
            self.server.pattern_backward = None

        print('duration of algorithm: {}s'.format(time.time() - start))
        return self

    def predict_proba_new(self, X, n_label):
        assert self.firing_graph is not None, "First fit firing graph"
        l_partitions = [p for p in sorted(self.firing_graph.partitions, key=lambda x: x['indices'][0])]
        n_label = len(set([p['label_id'] for p in self.firing_graph.partitions]))


        sax_activations = self.firing_graph\
            .reset_output(l_outputs=[p.get('group_id', 0) for p in self.firing_graph.partitions])\
            .propagate(X)
        import IPython
        IPython.embed()
        ax_probas = self.d_merger['C=0.1,fit_intercept=True,penalty=l2'].predict_proba(sax_activations.A)
        return ax_probas[:, [1]]



        # group_id in partition
        # sax_probas = self.firing_graph\
        #     .reset_output(l_outputs=[p.get('group_id', 0) for p in self.firing_graph.partitions])\
        #     .propagate_values(X, np.array([p['precision'] for p in l_partitions]))
        #
        # #ax_probas = sax_probas.max(axis=1).A[:, 0]
        # sax_coefs = csc_matrix((sax_probas > 0).A.cumsum(axis=1) * (sax_probas > 0).A)
        # sax_coefs.data = np.exp(-0 * (sax_coefs.data - 1))
        # ax_probas = sax_probas.multiply(sax_coefs).sum(axis=1).A / (sax_coefs.sum(axis=1).A + 10 + 1e-6)
        #
        # return ax_probas

    def predict_proba(self, X, n_label, min_probas=0.1):
        """

        :param X:
        :return:
        """
        assert self.firing_graph is not None, "First fit firing graph"
        l_partitions = [p for p in sorted(self.firing_graph.partitions, key=lambda x: x['indices'][0])]

        # group_id in partition
        sax_probas = self.firing_graph\
            .reset_output(l_outputs=[p.get('group_id', 0) * 2 + p['label_id'] for p in self.firing_graph.partitions])\
            .propagate_values(X, np.array([p['precision'] for p in l_partitions]))

        if n_label == 1:
            # TODO: True sparse
            return (sax_probas.sum(axis=1) / sax_probas.shape[1]).A
            # TODO: regul of 1
            #return (sax_probas.sum(axis=1) / ((sax_probas > 0).sum(axis=1) + 1)).A

        sax_sum = lil_matrix((sax_probas.shape[1], n_label), dtype=bool)
        for label in range(n_label):
            l_indices = [p.get('group_id', 0) * 2 + p['label_id'] for p in l_partitions if p['label_id'] == label]
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
