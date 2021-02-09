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
            split_drained_graph, single_select_amplified_bits, double_select_amplified_bits, compute_element_amplifier, \
            amplify_bits

        from matplotlib import pyplot as plt
        import time
        print('============== sample 1 ================')
        plot_path = 'DATA/test_new_paradigm/{}'
        ax_base_activations = X.sum(axis=0).A[0]

        ######################### STEP 1: AMPLIFY
        t = time.time()

        # Select random activation
        sample = X[0, :]

        # Init amplifier and initialize level to 5
        amplifier_fg = get_amplifier_firing_graph(sample.T, 5)
        sax_activations = amplifier_fg.propagate(X)
        sax_inner = sax_activations.astype(int).T.dot(X)
        print('round 0')
        print("Amplifier info: \n")
        print(f"activations: {sax_activations.sum(axis=0)}")

        # select bits amplified
        sax_I, level = single_select_amplified_bits(
            sax_inner, ax_base_activations, amplifier_fg.I.A[:, 0], mapping_feature_input, 0.5#, debug=True
        )
        print("amplificaiton done: \n")
        print(f"# feature {sax_I.astype(bool).T.dot(mapping_feature_input).sum()}, level: {level}")
        print(f'threshold selected: {round(level / sax_I.astype(bool).T.dot(mapping_feature_input).sum(), 2)}')

        print(f'============== Time amplifier is {round(time.time() - t, 2)} seconds')

        ########################### STEP 2: DRAIN
        t = time.time()

        # build drainer FG
        drainer_fg = get_drainer_firing_graph(sax_I, level)
        sax_activations = drainer_fg.propagate(X)
        target_prec = sax_activations.T.astype(int).dot(y).sum() / sax_activations.sum()
        drainer_params = self.__init_parameters(np.array([target_prec]))
        drainer_fg.matrices['Iw'] = sax_I * drainer_params.weights[0]

        print('round 0')
        print("Drainer info: \n")
        print(f'activation: {sax_activations.sum()}, precision: {target_prec}')

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
        print(f'============== Time drain / split is {round(time.time() - t, 2)} seconds')

        ######################### STEP 1: AMPLIFY'

        # Init amplifier and initialize level to 5
        amplifier_fg_left = get_amplifier_firing_graph(sax_I_left, drainer_fg.levels[0])
        amplifier_fg_right = get_amplifier_firing_graph(sax_I_right, drainer_fg.levels[0])
        sax_x_left, sax_x_right = amplifier_fg_left.propagate(X), amplifier_fg_right.propagate(X)
        sax_inner_left, sax_inner_right = sax_x_left.astype(int).T.dot(X), sax_x_right.astype(int).T.dot(X)

        print('round 1')
        print("Amplifier info: \n")
        print(f"activation left: {sax_x_left.sum(axis=0)}, activations right {sax_x_right.sum(axis=0)}")

        # Select bits amplified
        (sax_Il, levell), (sax_Ir, levelr) = double_select_amplified_bits(
            sax_inner_left, sax_inner_right, amplifier_fg_left.I.A[:, 0], amplifier_fg_right.I.A[:, 0],
            ax_base_activations, drainer_fg.levels[0], drainer_fg.levels[0], mapping_feature_input
        )

        # Print rest of useful information
        print(f"# feature left {sax_Il.astype(bool).T.dot(mapping_feature_input).sum()}, level left: {levell}")
        print(f'threshold selected: {round(levell / sax_Il.astype(bool).T.dot(mapping_feature_input).sum(), 2)}')
        print(f"sim left right {sax_Il.T.dot(sax_Ir)[0,0] / min(sax_Ir.sum(), sax_Il.sum())}")

        ########################### STEP 2: DRAIN'

        # build both drainer FG
        drainer_fgl = get_drainer_firing_graph(sax_Il, levell)
        drainer_fgr = get_drainer_firing_graph(sax_Ir, levelr)

        sax_activationl = drainer_fgl.propagate(X)
        target_precl = sax_activationl.T.astype(int).dot(y).sum() / sax_activationl.sum()
        sax_activationr = drainer_fgr.propagate(X)
        target_precr = sax_activationr.T.astype(int).dot(y).sum() / sax_activationr.sum()

        print('round 1')
        print("Drainer info: \n")
        print(f'activation left: {sax_activationl.sum()}, precision left: {target_precl}')
        print(f'activation right: {sax_activationr.sum()}, precision right: {target_precr}')

        # Continue with left child
        drainer_params = self.__init_parameters(np.array([target_precl]))
        drainer_fgl.matrices['Iw'] = sax_Il * drainer_params.weights[0]

        # split it using drained
        drained = FiringGraphDrainer(
            drainer_fgl, self.server, self.batch_size, **asdict(drainer_params.feedbacks)
        ) \
            .drain_all(n_max=self.draining_size) \
            .firing_graph

        sax_I_left, sax_I_right = split_drained_graph(
            drained.Iw, drained.backward_firing['i'], drainer_params.feedbacks.penalties,
            drainer_params.feedbacks.rewards, drainer_params.weights, mapping_feature_input
        )

        ######################### STEP 1: AMPLIFY''

        # Init amplifier and initialize level to 5
        amplifier_fg_left = get_amplifier_firing_graph(sax_I_left, drainer_fgl.levels[0])
        amplifier_fg_right = get_amplifier_firing_graph(sax_I_right, drainer_fgl.levels[0])
        sax_x_left, sax_x_right = amplifier_fg_left.propagate(X), amplifier_fg_right.propagate(X)
        sax_inner_left, sax_inner_right = sax_x_left.astype(int).T.dot(X), sax_x_right.astype(int).T.dot(X)

        print('round 2')
        print("Amplifier info: \n")
        print(f"activation left: {sax_x_left.sum(axis=0)}, activations right {sax_x_right.sum(axis=0)}")

        # Select bits amplified
        (sax_Il, levell), (sax_Ir, levelr) = double_select_amplified_bits(
            sax_inner_left, sax_inner_right, amplifier_fg_left.I.A[:, 0], amplifier_fg_right.I.A[:, 0],
            ax_base_activations, drainer_fgl.levels[0], drainer_fgl.levels[0], mapping_feature_input
        )

        # Print useful information
        print(f"# feature left {sax_Il.astype(bool).T.dot(mapping_feature_input).sum()}, level left: {levell}")
        print(f'threshold selected: {round(levell / sax_Il.astype(bool).T.dot(mapping_feature_input).sum(), 2)}')
        print(f"sim left right {sax_Il.T.dot(sax_Ir)[0,0] / min(sax_Ir.sum(), sax_Il.sum())}")

        ########################### STEP 2: DRAIN''

        # build both drainer FG
        drainer_fgl = get_drainer_firing_graph(sax_Il, levell)
        drainer_fgr = get_drainer_firing_graph(sax_Ir, levelr)

        sax_activationl = drainer_fgl.propagate(X)
        target_precl = sax_activationl.T.astype(int).dot(y).sum() / sax_activationl.sum()
        sax_activationr = drainer_fgr.propagate(X)
        target_precr = sax_activationr.T.astype(int).dot(y).sum() / sax_activationr.sum()

        print('round 2')
        print("Drainer info: \n")
        print(f'activation left: {sax_activationl.sum()}, precision left: {target_precl}')
        print(f'activation right: {sax_activationr.sum()}, precision right: {target_precr}')

        # Continue with left child
        drainer_params = self.__init_parameters(np.array([target_precl]))
        drainer_fgl.matrices['Iw'] = sax_Il * drainer_params.weights[0]

        # split it using drained
        drained = FiringGraphDrainer(
            drainer_fgl, self.server, self.batch_size, **asdict(drainer_params.feedbacks)
        ) \
            .drain_all(n_max=self.draining_size) \
            .firing_graph

        sax_I_left, sax_I_right = split_drained_graph(
            drained.Iw, drained.backward_firing['i'], drainer_params.feedbacks.penalties,
            drainer_params.feedbacks.rewards, drainer_params.weights, mapping_feature_input,
        )

        ######################### STEP 1: AMPLIFY'''

        # Init amplifier and initialize level to 5
        amplifier_fg_left = get_amplifier_firing_graph(sax_I_left, drainer_fgl.levels[0])
        amplifier_fg_right = get_amplifier_firing_graph(sax_I_right, drainer_fgl.levels[0])
        sax_x_left, sax_x_right = amplifier_fg_left.propagate(X), amplifier_fg_right.propagate(X)
        sax_inner_left, sax_inner_right = sax_x_left.astype(int).T.dot(X), sax_x_right.astype(int).T.dot(X)

        print('round 3')
        print("Amplifier info: \n")
        print(f"activation left: {sax_x_left.sum(axis=0)}, activations right {sax_x_right.sum(axis=0)}")

        # Select bits amplified
        (sax_Il, levell), (sax_Ir, levelr) = double_select_amplified_bits(
            sax_inner_left, sax_inner_right, amplifier_fg_left.I.A[:, 0], amplifier_fg_right.I.A[:, 0],
            ax_base_activations, drainer_fgl.levels[0], drainer_fgl.levels[0], mapping_feature_input
        )

        # Print useful information
        print(f"# feature left {sax_Il.astype(bool).T.dot(mapping_feature_input).sum()}, level left: {levell}")
        print(f'threshold selected: {round(levell / sax_Il.astype(bool).T.dot(mapping_feature_input).sum(), 2)}')
        print(f"sim left right {sax_Il.T.dot(sax_Ir)[0,0] / min(sax_Ir.sum(), sax_Il.sum())}")

        ########################### STEP 2: DRAIN'''

        # build both drainer FG
        drainer_fgl = get_drainer_firing_graph(sax_Il, levell)
        drainer_fgr = get_drainer_firing_graph(sax_Ir, levelr)

        sax_activationl = drainer_fgl.propagate(X)
        target_precl = sax_activationl.T.astype(int).dot(y).sum() / sax_activationl.sum()
        sax_activationr = drainer_fgr.propagate(X)
        target_precr = sax_activationr.T.astype(int).dot(y).sum() / sax_activationr.sum()

        print('round 3')
        print("Drainer info: \n")
        print(f'activation left: {sax_activationl.sum()}, precision left: {target_precl}')
        print(f'activation right: {sax_activationr.sum()}, precision right: {target_precr}')

        # Continue with left child
        drainer_params = self.__init_parameters(np.array([target_precl]))
        drainer_fgl.matrices['Iw'] = sax_Il * drainer_params.weights[0]

        # split it using drained
        drained = FiringGraphDrainer(
            drainer_fgl, self.server, self.batch_size, **asdict(drainer_params.feedbacks)
        ) \
            .drain_all(n_max=self.draining_size) \
            .firing_graph

        sax_I_left, sax_I_right = split_drained_graph(
            drained.Iw, drained.backward_firing['i'], drainer_params.feedbacks.penalties,
            drainer_params.feedbacks.rewards, drainer_params.weights, mapping_feature_input
        )

        ######################### STEP 1: AMPLIFY''''

        # Init amplifier and initialize level to 5
        amplifier_fg_left = get_amplifier_firing_graph(sax_I_left, drainer_fgl.levels[0])
        amplifier_fg_right = get_amplifier_firing_graph(sax_I_right, drainer_fgl.levels[0])
        sax_x_left, sax_x_right = amplifier_fg_left.propagate(X), amplifier_fg_right.propagate(X)
        sax_inner_left, sax_inner_right = sax_x_left.astype(int).T.dot(X), sax_x_right.astype(int).T.dot(X)

        print('round 4')
        print("Amplifier info: \n")
        print(f"activation left: {sax_x_left.sum(axis=0)}, activations right {sax_x_right.sum(axis=0)}")

        # Select bits amplified
        (sax_Il, levell), (sax_Ir, levelr) = double_select_amplified_bits(
            sax_inner_left, sax_inner_right, amplifier_fg_left.I.A[:, 0], amplifier_fg_right.I.A[:, 0],
            ax_base_activations, drainer_fgl.levels[0], drainer_fgl.levels[0], mapping_feature_input
        )

        # Print useful information
        print(f"# feature left {sax_Il.astype(bool).T.dot(mapping_feature_input).sum()}, level left: {levell}")
        print(f'threshold selected: {round(levell / sax_Il.astype(bool).T.dot(mapping_feature_input).sum(), 2)}')
        print(f"sim left right {sax_Il.T.dot(sax_Ir)[0,0] / min(sax_Ir.sum(), sax_Il.sum())}")

        ########################### STEP 2: DRAIN''''

        # build both drainer FG
        drainer_fgl = get_drainer_firing_graph(sax_Il, levell)
        drainer_fgr = get_drainer_firing_graph(sax_Ir, levelr)

        sax_activationl = drainer_fgl.propagate(X)
        target_precl = sax_activationl.T.astype(int).dot(y).sum() / sax_activationl.sum()
        sax_activationr = drainer_fgr.propagate(X)
        target_precr = sax_activationr.T.astype(int).dot(y).sum() / sax_activationr.sum()

        print('round 4')
        print("Drainer info: \n")
        print(f'activation left: {sax_activationl.sum()}, precision left: {target_precl}')
        print(f'activation right: {sax_activationr.sum()}, precision right: {target_precr}')

        # Continue with left child
        drainer_params = self.__init_parameters(np.array([target_precl]))
        drainer_fgl.matrices['Iw'] = sax_Il * drainer_params.weights[0]

        # split it usiamplifier_fg_left.propagate(X)ng drained
        drained = FiringGraphDrainer(
            drainer_fgl, self.server, self.batch_size, **asdict(drainer_params.feedbacks)
        ) \
            .drain_all(n_max=self.draining_size) \
            .firing_graph

        sax_I_left, sax_I_right = split_drained_graph(
            drained.Iw, drained.backward_firing['i'], drainer_params.feedbacks.penalties,
            drainer_params.feedbacks.rewards, drainer_params.weights, mapping_feature_input, debug=True, save=True
        )

        ######################### STEP 1: AMPLIFY'''''

        # Init amplifier and initialize level to 5
        amplifier_fg_left = get_amplifier_firing_graph(sax_I_left, drainer_fgl.levels[0])
        amplifier_fg_right = get_amplifier_firing_graph(sax_I_right, drainer_fgl.levels[0])
        sax_x_left, sax_x_right = amplifier_fg_left.propagate(X), amplifier_fg_right.propagate(X)

        print('round 5')
        print("Amplifier info: \n")
        print(f"activation left: {sax_x_left.sum(axis=0)}, activations right {sax_x_right.sum(axis=0)}")
        print('too few activation atomic state reached')
        #
        # # TODO: activations of both left and right are too small to be amplified. get Back to last amplified vertex and
        # #   analyze it
        # import IPython
        # IPython.embed()
        # level = sax_Il.astype(bool).T.dot(mapping_feature_input).sum() - 1
        # final_fg = get_amplifier_firing_graph(sax_Il, level)
        #
        # # Get some stats
        # sax_x_final = final_fg.propagate(X).tocsc()
        # sax_inner_final = sax_x_final.astype(int).T.dot(X)
        # prec = sax_x_final[:, 0].T.astype(int).dot(y).A / sax_x_final[:, 0].sum()
        # print(f"Final fg at level {level} is has prec {prec} and activate {sax_x_final[:, 0].sum()} times")
        #
        # for j in range(mapping_feature_input.shape[1]):
        #
        #     ax_inner_final_sub = sax_inner_final.A[:, mapping_feature_input.A[:, j]]
        #     ax_origin_mask_final = ~final_fg.I.A[:, 0][mapping_feature_input.A[:, j]]
        #     d_criterion_l, d_origin_signals_l, d_other_signals_l = compute_element_amplifier(
        #         ax_inner_final_sub, ax_origin_mask_final, ax_base_activations[mapping_feature_input.A[:, j]]
        #     )
        #
        #     fig, l_axes = plt.subplots(1, 3)
        #     print(d_criterion_l)
        #     # Plot details of origin bits
        #     l_axes[0].plot(d_origin_signals_l['bit_dist'], color="k")
        #     l_axes[0].plot(d_origin_signals_l['noise_dist'], '--', color="k")
        #     l_axes[0].plot(d_origin_signals_l['select'] * d_origin_signals_l['noise_dist'], 'o', color="k")
        #     l_axes[0].set_title(f'Origin dist {j} - amplifier')
        #
        #     # Plot details of other bits
        #     l_axes[1].plot(d_other_signals_l['bit_dist'], color="k")
        #     l_axes[1].plot(d_other_signals_l['noise_dist'], '--', color="k")
        #     l_axes[1].plot(d_other_signals_l['select'] * d_other_signals_l['noise_dist'], 'o', color="k")
        #     l_axes[1].set_title(f'Other dist {j} - amplifier')
        #
        #     # Plot details of all selcted bits
        #     l_axes[2].plot(d_other_signals_l['select'] + d_origin_signals_l['select'], color="k")
        #     l_axes[2].set_title(f'dist {j} of selected bits - amplifier')
        #     plt.show()
        #
        # # TODO: test random to see if we manage to randomly create such a specific shit
        # import IPython
        # IPython.embed()
        #
        # ax_bit_counts = drainer_fgl.I.T.dot(mapping_feature_input.astype(int)).A[0]
        # test_I = csc_matrix(drainer_fgl.I.shape)
        # for j in range(mapping_feature_input.shape[1]):
        #     n_bits = mapping_feature_input[:, j].sum()
        #     ax_rvalues = np.random.binomial(1, ax_bit_counts[j] / n_bits, n_bits)
        #     test_I[mapping_feature_input.A[:, j], 0] = ax_rvalues
        #
        # level = test_I.astype(bool).T.dot(mapping_feature_input).sum()
        # test_fg = YalaBasePatterns.from_input_matrix(
        #     test_I, [{'indices': 0, 'output_id': 0, 'label': 0, 'precision': 0}], np.array([level])
        # )

        ### We got them !!


        # TODO: now what would be a generic loop of the above example
        stop, l_inputs, l_levels, n_it = False, None, None, 0
        while not stop:
            print(n_it)
            ########## Amplify
            if l_inputs is None:
                l_inputs = [sample.T]
                l_levels = [5]

            l_amplifiers = [get_amplifier_firing_graph(sax_i, l_levels[i]) for i, sax_i in enumerate(l_inputs)]
            l_activations = [amp.propagate(X) for amp in l_amplifiers]

            print(l_amplifiers[0].propagate(X).sum(axis=0))
            if l_activations[0].sum() < 100:
                stop = True
                break

            current_final = l_amplifiers[0].copy()

            for i, sax_x in enumerate(l_activations[:1]):
                sax_inner = sax_x.astype(int).T.dot(X)
                sax_I, level = amplify_bits(
                    sax_inner, l_amplifiers[i].I.A[:, 0], ax_base_activations, l_levels[i], mapping_feature_input, 0.5
                )

                if n_it == 0:
                    level = 6

            print(sax_I.nnz, level)

            # This could be computed using amplifier
            drainer_fg = get_drainer_firing_graph(sax_I, level)
            sax_activations = drainer_fg.propagate(X)
            target_prec = sax_activations.T.astype(int).dot(y).sum() / sax_activations.sum()

            ########## Drain
            drainer_params = self.__init_parameters(np.array([target_prec]))
            drainer_fg.matrices['Iw'] = sax_I * drainer_params.weights[0]

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

            l_inputs = [sax_I_left]
            l_levels = [level]
            n_it += 1



        import IPython
        IPython.embed()

        level = current_final.I[:, 0].astype(bool).T.dot(mapping_feature_input).sum() - 1
        final_fg = get_amplifier_firing_graph(current_final.I[:, 0], level)
        sax_x_final = final_fg.propagate(X).tocsc()
        sax_inner_final = sax_x_final.astype(int).T.dot(X)
        prec = sax_x_final[:, 0].T.astype(int).dot(y).A / sax_x_final[:, 0].sum()
        print(f"Final fg at level {level} is has prec {prec} and activate {sax_x_final[:, 0].sum()} times")
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
