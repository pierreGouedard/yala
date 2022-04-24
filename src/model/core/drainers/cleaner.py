# Global import
import numpy as np
from scipy.sparse import csc_matrix
from copy import deepcopy as copy

# Local import
from src.model.utils import sample_from_mask
from src.model.core.patterns import YalaBasePatterns
from src.model.core.data_models import FgComponents
from src.model.core.drainers.visualizer import Visualizer


class Cleaner(Visualizer):
    """Expander"""
    def __init__(
            self, server, sax_bf_map, drainer_params, min_firing=100, n_min_bounds=3, perf_plotter=None,
    ):
        # call parent constructor
        self.n_min_bounds = n_min_bounds
        self.ax_cleaned_features = None
        super().__init__(server, sax_bf_map, drainer_params, min_firing, level_delta=1, perf_plotter=perf_plotter)

    def select_support_bits(self, sax_drained_weights, sax_count_activations):
        # Get selected bits keep if support
        sax_active_inputs = self.select_inputs(sax_drained_weights, sax_count_activations)
        sax_drained_inputs = sax_count_activations > 0

        # Compute input cardinality for each features
        ax_card_selected: np.ndarray = self.b2f(sax_active_inputs.astype(int)).A
        ax_card_signal: np.ndarray = self.b2f(sax_drained_inputs.astype(int)).A

        # Identify bounds to remove
        ax_mask_selected = ax_card_selected < ax_card_signal

        # Visualize result of cleaning
        if self.advanced_plot_perf_enabled:
            self.visualize_multi_selection(sax_active_inputs, ax_mask_selected)

        # Keep a minimum number of bounds
        ax_sampled = sample_from_mask(
            self.b2f(self.fg_mask.I).A & (~ax_mask_selected),
            ax_n=np.maximum(self.n_min_bounds - ax_mask_selected.sum(axis=1), 0)
        )
        ax_mask_selected |= ax_sampled

        # Build support inputs
        sax_support_bits = (sax_active_inputs + self.pre_draining_inputs).multiply(
            self.f2b(csc_matrix(ax_mask_selected.T))
        ) + self.pre_draining_inputs

        # Keep only selected bounds
        sax_support_bits = sax_support_bits.multiply(self.f2b(csc_matrix(ax_mask_selected.T)))

        # Track cleaned features
        self.ax_cleaned_features = self.b2f(self.fg_mask.I).A.T ^ self.b2f(sax_support_bits).A.T
        import IPython
        IPython.embed()
        return sax_support_bits

    def reset(self):
        self.ax_cleaned_features = None
        Visualizer.reset(self)

    def build_patterns(self, component, **kwargs):
        # Build convex hull
        sax_hull_inputs = self.build_convex_hull(component, kwargs['x'], kwargs['fg'])

        # Build components
        fg_comp, mask_comp = self.build_components(component, sax_hull_inputs)
        self.pre_draining_inputs = component.inputs

        # Create mask pattern from comp
        self.fg_mask = YalaBasePatterns.from_fg_comp(mask_comp)
        self.firing_graph = YalaBasePatterns.from_fg_comp(fg_comp)

    def build_components(self, comp, sax_hull_inputs):
        fg_comp = FgComponents(
            inputs=csc_matrix(comp.inputs.A ^ self.f2b(self.b2f(comp.inputs).T).A).multiply(sax_hull_inputs),
            partitions=copy(comp.partitions), levels=np.ones(len(comp))
        )

        return fg_comp, comp.copy(levels=copy(comp.levels) - 1)

    def build_convex_hull(self, component, sax_x, sax_fg):
        # Get masked activations
        sax_product = sax_x.astype(bool).T.dot(sax_fg)

        # Compute feature masks
        ax_card, ax_mask_component = self.b2f(sax_product.astype(int)).A, self.b2f(component.inputs.astype(bool)).A
        ax_mask = (0 < ax_card) & (ax_card < self.get_card_features(ax_card.shape[0])) & ax_mask_component

        return sax_product.multiply(self.f2b(csc_matrix(ax_mask.T)))

