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
from src.model.helpers.amplifier import Amplifier

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

    def init_parameters(self, ax_precision):
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
            split_drained_graph, final_bit_selection, amplify_bits, show_significance_plot, \
            show_activation_stats, show_draining_stats, show_diff_fg, reselect_graph
        from scipy.ndimage.interpolation import shift
        from matplotlib import pyplot as plt
        import time
        print('============== sample 1 ================')
        plot_path = 'DATA/test_new_paradigm/{}'
        ax_base_activations = X.sum(axis=0).A[0]
        ax_samples = np.random.randint(0, X.shape[0], 10)
        l_fg = []
        for ind in ax_samples:

            sample = X[ind, :]
            print(ind)
            self.server.stream_features()
            ax_thresh = np.ones(mapping_feature_input.shape[1]) * 0.5
            stop, do_amplify, sax_inputs, level, n_it = False, True, sample.T, 5, 0
            while not stop:
                print(f"Iteration : {n_it}")

                ########## Amplify
                amplifier_fg = get_amplifier_firing_graph(sax_inputs, level)
                sax_activations = amplifier_fg.propagate(X)

                # Print useful information
                print("Amplifier info: \n")
                print(f"Before amplification: #Activations: {sax_activations.sum(axis=0)}")

                if sax_activations.tocsc()[:, 0].sum() < 100 or not do_amplify:
                    # Final amplification
                    sax_I, level, ax_thresh, stop = final_bit_selection(
                        current_fg, mapping_feature_input, X, ax_base_activations, ax_thresh
                    )
                    if stop:
                        final_fg = YalaBasePatterns.from_input_matrix(
                            sax_I, [{'indices': 0, 'output_id': 0, 'label': 0, 'precision': 0}], np.array([level])
                        )
                        break

                else:

                    sax_I, level, ax_thresh = amplify_bits(
                        sax_activations.astype(int).T.dot(X), amplifier_fg.I.A[:, 0], ax_base_activations,
                        mapping_feature_input, ax_thresh
                    )

                    if n_it == 0:
                        level = 6

                ax_thresh = ax_thresh.clip(max=0.5, min=0.5)
                current_fg = get_amplifier_firing_graph(sax_I, sax_I[:, 0].astype(bool).T.dot(mapping_feature_input).sum())

                # Print useful information
                print(f"After amplification: # Features: {sax_I[:, 0].astype(bool).T.dot(mapping_feature_input).sum()}, Level: {level}")

                # This could be computed using amplifier
                drainer_fg = get_drainer_firing_graph(sax_I, level, mapping_feature_input, n_drain=12)#mapping_feature_input.shape[1])
                sax_activations = drainer_fg.propagate(X)
                target_prec = sax_activations.T.astype(int).dot(y).sum() / sax_activations.sum()

                do_amplify = True
                if target_prec > self.picker_params['max_precision']:
                    sax_inputs, do_amplify = sax_I.copy(), False
                    continue

                ########## Drain
                drainer_params = self.init_parameters(np.array([target_prec]))
                drainer_fg.matrices['Iw'] = sax_I * drainer_params.weights[0]
                print("Drainer info: \n")
                print(f'Before draining: # activation drainer: {sax_activations.sum()}, target prec: {target_prec}')

                # split it using drained
                import time
                t = time.time()
                drained = FiringGraphDrainer(
                    drainer_fg, self.server, self.batch_size, **asdict(drainer_params.feedbacks)
                ) \
                    .drain_all(n_max=self.draining_size) \
                    .firing_graph

                sax_input_l, sax_input_r = split_drained_graph(drained.Iw, drained.backward_firing['i'])
                sax_inputs = sax_input_l if bool(np.random.binomial(1, 0.8)) else sax_input_r
                n_it += 1
                print(time.time() - t)
            l_fg.append(final_fg)




        import IPython
        IPython.embed()

        # Analysis prec
        show_draining_stats(l_fg[0], mapping_feature_input, X, y)

        # Analysis activation
        for fg in l_fg:
            show_activation_stats(fg, mapping_feature_input, X, y)

        # Analysis compare against lucky sampling
        for fg in l_fg:
            show_significance_plot(fg, mapping_feature_input, X)

        ax_cov_inputs = np.zeros((len(l_fg), len(l_fg)))
        ax_cov_signals = np.zeros((len(l_fg), len(l_fg)))
        sax_x = csc_matrix(y.shape, dtype=bool)
        for i, fga in enumerate(l_fg):
            sax_xa = fga.propagate(X)
            sax_x += sax_xa.tocsc()
            for j, fgb in enumerate(l_fg):
                if i > j:
                    continue

                ax_cov_inputs[i, j] = fga.I.T.astype(int).dot(fgb.I)[0, 0]
                sax_xb = fgb.propagate(X)
                ax_cov_signals[i, j] = sax_xa.T.astype(int).dot(sax_xb)[0, 0]
        # END


        # TODO: implement almost production level code of the new routine
        #   * Vectorize as much as possible
        #   * Based on amplifier and Drainer => limit as much as possible the utils function
        #   * May be at some point, externalize the amplifier, as the drainer.
        #   * As previously planned take care of the encoding in here, so that yala can respect scikit-learn interface
        #       and can be added to my (to release) DS platform for personal project based on Kedro / scikit / keras
        #   * Work on Portfolio management project :)

        # TODO: Schedule of prod implementation of new paradigm:
        #   * 1. Create amplifier class, generalize and vectorize its operations
        #   * 2. Create util codes that implement necessary operations to go from drain to amplify and amplify to drain
        #   * 3. implement final selection of vertices:
        #       3.A. Remove duplicated factors
        #       3.B. Create reducer (mean, regul mean, logistic reg)

        # TODO: Post usability implementation:
        #  Make a true fucking grid search, get perf. The continuation of this project doesn't depends on the performace
        #  found on the kaggle dataset. Just continue making it a prod level algo and avoid making change to the current
        #  algorithm. That is: Take care of encoding and unit test and epuration of the code (mypy, flake8, Black)


        #### THERE WE GO FOR THE FINAL FORM OF THE ALGORITHM
        init_level, init_partition = 5, [{'indices': 0, 'output_id': 0, 'label': 0, 'precision': 0}]
        amplifier = Amplifier(self.server, mapping_feature_input, self.draining_size)

        for i in range(self.max_iter):
            print("[YALA]: Iteration {}".format(i))

            self.amplifier.sample_and_amplify()

            self.drainer_params = self.init_parameters(self.server.get_init_precision())
            firing_graph = build_draining_firing_graph(self.sampler, self.drainer_params)
            stop = False
            while not stop:

                firing_graph = prepare_draining_graph()
                # Drain firing graph
                firing_graph = FiringGraphDrainer(
                    firing_graph, self.server, self.batch_size, **asdict(self.drainer_params.feedbacks)
                ) \
                    .drain_all(n_max=self.draining_size) \
                    .firing_graph

                # Pick partial patterns
                partials, self.drainer_params = self.picker.pick_patterns_multi_label(self.server, firing_graph)

                # Update stop criteria
                stop = partials is None
                if not stop:
                    firing_graph = build_draining_firing_graph(self.sampler, self.drainer_params, partials)

        #### END
        # Core loop
        import time
        start = time.time()
        count_no_update, l_dropouts = 0, []
        for i in range(self.max_iter):
            print("[YALA]: Iteration {}".format(i))

            # infer init params from signal and build initial graph
            print(self.server.get_init_precision())
            self.drainer_params = self.init_parameters(self.server.get_init_precision())
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

            #### TEST
            level = self.picker.completes.I[:, 0].T.dot(mapping_feature_input).sum()
            fg = get_drainer_firing_graph(self.picker.completes.matrices['Iw'][:, 0], level, mapping_feature_input)
            fg = get_amplifier_firing_graph(self.picker.completes.matrices['Iw'][:, 0], level)
            import matplotlib.pyplot as plt
            import IPython
            IPython.embed()
            sax_I, level = reselect_graph(fg, mapping_feature_input, X)

            # Show inputs
            for i in range(mapping_feature_input.shape[1]):
                ax_in = fg.I.A[mapping_feature_input.A[:, i], 0]
                plt.plot(ax_in)
                plt.title(f'input feature {i}')
                plt.show()

            show_draining_stats(fg, mapping_feature_input, X, y)
            #### END TEST

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
