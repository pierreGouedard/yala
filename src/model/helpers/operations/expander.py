# Global import
import numpy as np
from scipy.sparse import diags, csc_matrix

# Local import
from src.model.helpers.patterns import YalaBasePatterns, YalaTopPattern
from src.model.helpers.data_models import FgComponents
from src.model.helpers.operations.drainer import YalaDrainer


class Expander(YalaDrainer):
    """Expander"""

    def __init__(self, server, sax_bf_map, drainer_params, min_firing=100, perf_plotter=None, plot_perf_enabled=False):
        # call parent constructor
        super().__init__(server, sax_bf_map, drainer_params, min_firing, perf_plotter, plot_perf_enabled)

    def select_inputs(self, sax_weight, sax_count):
        # TODO: refactor (maybe)
        ax_p, ax_r = self.drainer_params.feedbacks.get_all()
        ax_w, ax_target_prec = self.drainer_params.weights, self.drainer_params.get_limit_precisions()

        # Get input weights and count
        sax_mask = (sax_weight > 0).multiply(sax_count > 0)

        sax_nom = sax_weight.multiply(sax_mask) - sax_mask.dot(diags(ax_w, format='csc'))
        sax_denom = sax_mask.multiply(sax_count.dot(diags(ax_p + ax_r, format='csc')))
        sax_precision = sax_nom.multiply(sax_denom.astype(float).power(-1))
        sax_precision += (sax_precision != 0).dot(diags(ax_p / (ax_p + ax_r), format='csc'))

        return sax_precision > (sax_precision > 0).dot(diags(ax_target_prec, format='csc'))

    def merge_inputs(self, sax_mask_inputs, sax_drained_weights, sax_count_activations):
        # TODO: refactor

        # Get bits with limit_precision
        sax_drained_inputs = self.select_inputs(sax_drained_weights, sax_count_activations)

        # Get new candidate features and their bits cardinality
        ax_mask_features = ~sax_mask_inputs.T.dot(self.bf_map).A
        ax_card_features = self.bf_map.sum(axis=0).A[[0] * ax_mask_features.shape[0], :] * ax_mask_features

        # Get cardinality of each drained feature's bits
        ax_card_selected = sax_drained_inputs.astype(int).T.dot(self.bf_map).A * ax_mask_features

        # Choose new candidate features (cardinality above 0 and lower than feature cardinality)
        ax_mask_selected = self.sample_from_mask((ax_card_selected < ax_card_features) * (0 < ax_card_selected))
        sax_mask_new = self.bf_map.dot(csc_matrix(ax_mask_selected.T))

        # Compute new inputs from refined existing bits and new candidates bits
        sax_inputs = sax_drained_inputs.multiply(sax_mask_inputs) + sax_drained_inputs.multiply(sax_mask_new)

        return sax_inputs

    def select(self):
        # TODO: refactor

        # Compute new inputs, levels and partitions
        sax_inputs = self.merge_inputs(self.fg_mask.I, self.firing_graph.Iw, self.firing_graph.backward_firing['i'])
        ax_levels = sax_inputs.T.dot(self.bf_map).A.sum(axis=1)
        l_partitions = [{**p, "precision": None} for p in self.fg_mask.partitions]

        # Create component
        fg_comp = FgComponents(inputs=sax_inputs, partitions=l_partitions, levels=ax_levels)

        if self.plot_perf_enabled:
            self.visualize_fg(YalaBasePatterns.from_fg_comp(fg_comp))

        return fg_comp

    def build_patterns(self, component):
        # TODO: refactor

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
