# Global import
import numpy as np
from scipy.sparse import csc_matrix

# Local import
from src.model.core.drainers.visualizer import Visualizer


class Shaper(Visualizer):
    """Shaper"""

    def __init__(
            self, server, bitmap, drainer_params, min_firing=100, min_bounds=2, perf_plotter=None,
    ):
        # call parent constructor
        super().__init__(server, bitmap, drainer_params, min_firing, min_bounds, perf_plotter=perf_plotter)

    def select_support_bits(self, sax_drained_weights, sax_count_activations):
        # TODO: appy fucking rules + may be do something so that the integrity of the base is preserved (no hole)
        # Get active & drained bits
        sax_active_inputs = self.select_inputs(sax_drained_weights, sax_count_activations)
        sax_drained_inputs = sax_count_activations > 0

        # Compute input cardinality for each features
        ax_card_selected: np.ndarray = self.bitmap.b2f(sax_active_inputs.astype(int)).A
        ax_card_signal: np.ndarray = self.bitmap.b2f(sax_drained_inputs.astype(int)).A

        # Identify bounds to select
        ax_mask_selected = ax_card_selected < ax_card_signal

        # Build support inputs
        sax_support_bits = (sax_active_inputs + self.original_inputs).multiply(
            self.bitmap.f2b(csc_matrix(ax_mask_selected.T))
        )

        # Mask's bound that have not been selected
        sax_mask_mask = self.bitmap.f2b(csc_matrix(self.bitmap.b2f(self.fg_mask.I).A ^ ax_mask_selected, dtype=bool).T)
        sax_support_bits += self.fg_mask.I.multiply(sax_mask_mask)

        # Visualize result of shaper
        if self.advanced_plot_perf_enabled:
            self.visualize_multi_selection(sax_active_inputs, ax_mask_selected)
        import IPython
        IPython.embed()
        return sax_support_bits

