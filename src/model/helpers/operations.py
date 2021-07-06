# Global import
import numpy as np
from dataclasses import asdict
from itertools import groupby
from firing_graph.solver.drainer import FiringGraphDrainer
from scipy.sparse import diags, csc_matrix

# Local import
from src.model.utils import init_parameters
from src.model.helpers.patterns import YalaBasePatterns, YalaTopPattern
from src.model.helpers.data_models import FgComponents


class Refiner(FiringGraphDrainer):
    def __init__(
            self, server, sax_bf_map, drainer_params, min_firing=100, n_update=10, f_candidates=None, perf_plotter=None,
            plot_perf_enabled=False
    ):
        # map bit to features and candidate features
        self.bf_map, self.mask_candidate = sax_bf_map, None
        self.update_candidate_mask(f_candidates)

        # Parameters for draining
        self.drainer_params = drainer_params

        # complement attributes
        self.min_firing, self.n_update = min_firing, n_update

        # Optional callable to visualize perf
        self.perf_plotter = perf_plotter
        self.plot_perf_enabled = plot_perf_enabled

        # call parent constructor
        super().__init__(None, server, self.drainer_params.batch_size)

    def visualize_fg(self, firing_graph):
        # Get masked activations
        sax_x = self.server.get_sub_forward(self.perf_plotter.indices)
        ax_yhat = firing_graph.propagate(sax_x).A

        # Plot perf viz
        self.perf_plotter(ax_yhat)

    def update_candidate_mask(self, f_candidates=None):
        if f_candidates is not None:
            self.mask_candidate = csc_matrix([[
                True if i in f_candidates else False for i in range(self.bf_map.shape[1])
            ]], dtype=bool)

        else:
            self.mask_candidate = csc_matrix(np.ones((1, self.bf_map.shape[1]), dtype=bool))

    def prepare(self, component):
        # Update precision and drainer params
        self.update_precision(component)
        self.update_drainer_params(component)

        # Build top and bottom patterns
        self.build_patterns(component)

        return self

    def drain_all(self, **kwargs):
        # Update top and backward pattern of server
        gr = groupby([(p['label_id'], i) for i, p in enumerate(self.firing_graph.partitions)], key=lambda t: t[0])
        self.server.pattern_backward = YalaTopPattern.from_mapping({k: list(map(lambda x: x[1], v)) for k, v in gr})

        # FOR TEST: viz before
        if self.plot_perf_enabled:
            self.visualize_fg(self.fg_mask)

        # Drain
        super().drain_all(n_max=self.drainer_params.total_size)

        # Reset top pattern of server
        self.server.pattern_backward = None

        return self

    def select_inputs(self, sax_weight, sax_count):

        ax_p, ax_r = self.drainer_params.feedbacks.get_all()
        ax_w, ax_target_prec = self.drainer_params.weights, self.drainer_params.get_target_precisions()

        # Get input weights and count
        sax_mask = (sax_weight > 0).multiply(sax_count > 0)

        sax_nom = sax_weight.multiply(sax_mask) - sax_mask.dot(diags(ax_w, format='csc'))
        sax_denom = sax_mask.multiply(sax_count.dot(diags(ax_p + ax_r, format='csc')))
        sax_precision = sax_nom.multiply(sax_denom.astype(float).power(-1))
        sax_precision += (sax_precision != 0).dot(diags(ax_p / (ax_p + ax_r), format='csc'))

        import IPython
        IPython.embed()

        return sax_precision > 0

    def merge_inputs(self, sax_mask_inputs, sax_drained_weights, sax_count_activations):

        sax_drained_inputs = self.select_inputs(sax_drained_weights, sax_count_activations)

        # Get new candidate features and their bits cardinality
        ax_mask_features = ~sax_mask_inputs.T.dot(self.bf_map).A
        ax_card_features = self.bf_map.sum(axis=0).A[[0] * ax_mask_features.shape[0], :] * ax_mask_features

        # Get cardinality of each drained feature's bits
        ax_card_selected = sax_drained_inputs.astype(int).T.dot(self.bf_map).A * ax_mask_features

        # Choose new candidate features (cardinality above 0 and lower than feature cardinality)
        ax_mask_selected = (ax_card_selected < ax_card_features) * (0 < ax_card_selected)

        # TODO: add random selection with self.n_update

        sax_mask_new = self.bf_map.dot(csc_matrix(ax_mask_selected.T))

        # Compute new inputs from refined existing bits and new candidates bits
        sax_inputs = sax_drained_inputs.multiply(sax_mask_inputs) + sax_drained_inputs.multiply(sax_mask_new)

        return sax_inputs

    def select(self):

        # Compute new inputs, levels and partitions
        sax_inputs = self.merge_inputs(self.fg_mask.I, self.firing_graph.Iw, self.firing_graph.backward_firing['i'])
        ax_levels = sax_inputs.T.dot(self.bf_map).A.sum(axis=1)
        l_partitions = [{**p, "precision": None} for p in self.fg_mask.partitions]

        # Create component
        fg_comp = FgComponents(inputs=sax_inputs, partitions=l_partitions, levels=ax_levels)

        import IPython
        IPython.embed()

        if self.plot_perf_enabled:
            self.visualize_fg(YalaBasePatterns.from_fg_comp(fg_comp))

        for i in range(self.bf_map.shape[1]):
            sax_inputs_test = sax_inputs[:, 0].multiply(self.bf_map[:, i])
            if sax_inputs_test.nnz > 0:
                ax_levels_test = sax_inputs_test.T.dot(self.bf_map).A.sum(axis=1)
                l_partitions_tests = [{**p, "precision": None} for p in self.fg_mask.partitions]
                fg_comp_test = FgComponents(inputs=sax_inputs_test, partitions=l_partitions_tests, levels=ax_levels_test)
                self.visualize_fg(YalaBasePatterns.from_fg_comp(fg_comp_test))
                import IPython
                IPython.embed()

        return fg_comp

    def reset(self):
        self.reset_all()
        self.firing_graph, self.fg_mask = None, None

    def update_precision(self, component):

        # Get masked activations
        sax_x = self.server.next_forward(n=self.drainer_params.batch_size, update_step=False).sax_data_forward
        sax_y = self.server.next_backward(n=self.drainer_params.batch_size, update_step=False).sax_data_backward

        # Compute precision of each vertex and update partitions
        sax_x = YalaBasePatterns.from_fg_comp(component).propagate(sax_x)

        ax_precisions = (sax_y.T.astype(int).dot(sax_x) / (sax_x.sum(axis=0) + 1e-6)).A[0]
        component.partitions = [{**p, 'precision': ax_precisions[i]} for i, p in enumerate(component.partitions)]
        self.drainer_params.precisions = ax_precisions

    def update_drainer_params(self, component):

        # Get precision from component
        ax_precisions = np.array([p['precision'] for p in component.partitions])

        # Compute and update feedbacks and weights used in draining
        self.drainer_params.feedbacks, self.drainer_params.weights = \
            init_parameters(ax_precisions, self.drainer_params.margin, self.min_firing)
        self.update_pr(**asdict(self.drainer_params.feedbacks))

    def build_patterns(self, component):

        # Create top pattern from comp
        self.fg_mask = YalaBasePatterns.from_fg_comp(component)

        # Create base bottom comp
        self.firing_graph = YalaBasePatterns.from_fg_comp(component)

        # Augment with unexplored candidate features
        sax_candidates = self.bf_map.dot(self.mask_candidate[[0] * len(component), :].T) - \
            self.bf_map.dot(self.firing_graph.I.T.dot(self.bf_map).T)
        self.firing_graph.matrices['Iw'] += sax_candidates.astype(self.firing_graph.matrices['Iw'].dtype)

        # Update levels
        self.firing_graph.levels = np.ones(len(component))

        # Update input matrice's weights
        sax_weights = diags(self.drainer_params.weights, format='csc', dtype=self.firing_graph.matrices['Iw'].dtype)
        self.firing_graph.matrices['Iw'] = self.firing_graph.matrices['Iw'].dot(sax_weights)

        # Update mask draining
        self.firing_graph.matrices['Im'] = self.firing_graph.I

        # Update firing graph from parent
        self.reset_all()
