# Global import
import numpy as np
from scipy.sparse import csc_matrix

# Local import
from src.model.core.sampling import sample_from_mask
from src.model.core.drainers.visualizer import Visualizer


class Cleaner(Visualizer):
    """Expander"""
    def __init__(
            self, server, bitmap, drainer_params, min_firing=100, min_bounds=2, perf_plotter=None,
    ):
        # call parent constructor
        super().__init__(server, bitmap, drainer_params, min_firing, min_bounds, perf_plotter=perf_plotter)

    def select_support_bits(self, sax_drained_weights, sax_count_activations, **kwargs):
        # TODO: obsolete => Good but meant to remove 1 bound at a time
        # Get selected bits keep if support
        sax_active_inputs = self.select_inputs(sax_drained_weights, sax_count_activations)
        sax_drained_inputs = sax_count_activations > 0

        # Compute input cardinality for each features
        ax_card_selected: np.ndarray = self.bitmap.b2f(sax_active_inputs.astype(int)).A
        ax_card_signal: np.ndarray = self.bitmap.b2f(sax_drained_inputs.astype(int)).A

        # Identify bounds to remove
        ax_mask_selected = ax_card_selected < ax_card_signal

        # Visualize result of cleaning
        if self.advanced_plot_perf_enabled:
            self.visualize_multi_selection(sax_active_inputs, ax_mask_selected)

        # Keep a minimum number of bounds
        ax_sampled = sample_from_mask(
            self.bitmap.b2f(self.fg_mask.I).A & (~ax_mask_selected),
            ax_n=np.maximum(self.min_bounds - ax_mask_selected.sum(axis=1), 0)
        )
        ax_mask_selected |= ax_sampled

        # Build support inputs
        sax_support_bits = (sax_active_inputs + self.original_inputs).multiply(
            self.bitmap.f2b(csc_matrix(ax_mask_selected.T))
        ) + self.original_inputs

        # Keep only selected bounds
        sax_support_bits = sax_support_bits.multiply(self.bitmap.f2b(csc_matrix(ax_mask_selected.T)))

        return sax_support_bits

    def reset(self):
        Visualizer.reset(self)

