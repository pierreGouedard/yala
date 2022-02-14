# Global import
import numpy as np
from scipy.sparse import csc_matrix
from copy import deepcopy as copy

# Local import
from src.model.utils import sample_from_mask
from src.model.helpers.patterns import YalaBasePatterns
from src.model.helpers.data_models import FgComponents
from src.model.helpers.operations.visualizer import Visualizer


class Cleaner(Visualizer):
    """Expander"""
    def __init__(
            self, server, sax_bf_map, drainer_params, min_firing=100, n_min_bounds=3, perf_plotter=None,
            plot_perf_enabled=True, advanced_plot_perf_enabled=True
    ):
        # call parent constructor
        self.n_min_bounds = n_min_bounds
        self.advanced_plot_perf_enabled = advanced_plot_perf_enabled
        super().__init__(server, sax_bf_map, drainer_params, min_firing, perf_plotter, plot_perf_enabled)

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
            ax_card_signal > 0 & ~ax_mask_selected, ax_n=min(self.n_min_bounds - ax_mask_selected.sum(axis=1))
        )
        ax_mask_selected |= ax_sampled

        # Compute new inputs and metrics
        sax_support_bits = self.pre_draining_fg.I.multiply(self.f2b(csc_matrix(ax_mask_selected.T)))

        return sax_support_bits

    def build_patterns(self, component, **kwargs):
        # Build components
        fg_comp, mask_comp = self.build_components(component)

        # Create mask pattern from comp
        self.fg_mask = YalaBasePatterns.from_fg_comp(mask_comp)
        self.firing_graph = YalaBasePatterns.from_fg_comp(fg_comp)
        self.pre_draining_fg = YalaBasePatterns.from_fg_comp(component)

    def build_components(self, comp):
        fg_comp = FgComponents(
            inputs=csc_matrix(comp.inputs.A ^ self.f2b(self.b2f(comp.inputs).T).A),
            partitions=copy(comp.partitions), levels=np.ones(len(comp))
        )

        return fg_comp, comp.copy(levels=copy(comp.levels) - 1)


