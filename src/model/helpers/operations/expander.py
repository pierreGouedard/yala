# Global import
import numpy as np
from scipy.sparse import diags, csc_matrix

# Local import
from src.model.helpers.patterns import YalaBasePatterns
from src.model.helpers.data_models import FgComponents
from src.model.helpers.operations.drainer import YalaDrainer


class Expander(YalaDrainer):
    """Expander"""

    def __init__(self, server, sax_bf_map, drainer_params, min_firing=100, perf_plotter=None, plot_perf_enabled=False):
        # call parent constructor
        super().__init__(server, sax_bf_map, drainer_params, min_firing, perf_plotter, plot_perf_enabled)

    def select(self):

        # Compute new inputs, levels and partitions
        sax_inputs = self.merge_inputs(self.fg_mask.I, self.firing_graph.Iw, self.firing_graph.backward_firing['i'])
        ax_levels = sax_inputs.T.dot(self.bf_map).A.sum(axis=1)
        l_partitions = [{**p, "precision": None} for p in self.fg_mask.partitions]

        # Create component
        fg_comp = FgComponents(inputs=sax_inputs, partitions=l_partitions, levels=ax_levels)

        if self.plot_perf_enabled:
            self.visualize_fg(YalaBasePatterns.from_fg_comp(fg_comp))

        return fg_comp

    def merge_inputs(self, sax_mask_inputs, sax_drained_weights, sax_count_activations):
        # Get selected bits and and merge with base bit
        sax_drained_inputs = self.select_inputs(sax_drained_weights, sax_count_activations)
        sax_merged_inputs = sax_mask_inputs + sax_drained_inputs

        # Selected feature based on feature cardinality
        ax_card_features = self.bf_map.sum(axis=0).A[[0] * sax_merged_inputs.shape[1], :]
        ax_card_selected = sax_merged_inputs.astype(int).T.dot(self.bf_map).A
        ax_mask_selected = (ax_card_selected < ax_card_features) * (0 < ax_card_selected)

        # Compute new inputs from mask of feature
        sax_inputs = sax_merged_inputs.multiply(self.bf_map.dot(csc_matrix(ax_mask_selected.T)))

        return sax_inputs

    def build_patterns(self, component):
        # Create mask pattern from comp
        mask_component = component.copy()
        mask_component.levels = mask_component.levels - 1
        self.fg_mask = YalaBasePatterns.from_fg_comp(mask_component)

        # Create base bottom comp
        sax_mask = self.bf_map.dot(self.bf_map.T.dot(component.inputs))
        base_component = component.complement(sax_mask)
        base_component.levels = np.ones(len(base_component))
        self.firing_graph = YalaBasePatterns.from_fg_comp(base_component)

        # Update base matrice input's weights
        sax_weights = diags(self.drainer_params.weights, format='csc', dtype=self.firing_graph.matrices['Iw'].dtype)
        self.firing_graph.matrices['Iw'] = self.firing_graph.matrices['Iw'].dot(sax_weights)

        # Update mask draining
        self.firing_graph.matrices['Im'] = self.firing_graph.I

        # Update firing graph from parent
        self.reset_all()
