# Global import
import numpy as np
from scipy.sparse import csc_matrix

# Local import
from src.model.core.data_models import FgComponents
from src.model.core.drainers.visualizer import Visualizer
from src.model.utils import sample_from_mask


class Shaper(Visualizer):
    """Shaper"""

    def __init__(
            self, server, bitmap, drainer_params, min_firing=100, min_bounds=2, perf_plotter=None,
    ):
        # call parent constructor
        super().__init__(server, bitmap, drainer_params, min_firing, min_bounds, perf_plotter=perf_plotter)

    def select(self, merge=False):

        # Select support bits and mask bits
        sax_support_bits = self.select_support_bits(
            self.firing_graph.Iw, self.firing_graph.backward_firing['i'], keep_min_bounds=not merge
        )

        if not merge:
            base_component = FgComponents(
                inputs=sax_support_bits, partitions=self.firing_graph.partitions, levels=None
            )
            mask_comp = self.fg_mask.to_comp()

        else:
            base_component = FgComponents(
                inputs=sax_support_bits + self.fg_mask.I, partitions=self.firing_graph.partitions,
                levels=None
            )
            mask_comp = None

        base_component = base_component.update(partitions=self.update_partition_metrics(base_component))

        return base_component, mask_comp

    def select_support_bits(self, sax_drained_weights, sax_count_activations, keep_min_bounds=False):
        # Get active & drained bits
        sax_active_inputs = self.select_inputs(sax_drained_weights, sax_count_activations)

        # TODO: Add layer so that the integrity & convexity of base is preserved (no hole)

        # Compute input cardinality for each features
        ax_card_selected: np.ndarray = self.bitmap.b2f(sax_active_inputs.astype(int)).A
        ax_card_signal: np.ndarray = self.bitmap.b2f((sax_count_activations > 0).astype(int)).A

        # Identify bounds to select
        ax_mask_support = ax_card_selected < ax_card_signal

        # Keep a minimum number of bounds (if specified)
        if keep_min_bounds:
            ax_sampled = sample_from_mask(
                self.bitmap.b2f(self.original_inputs.astype(bool)).A & (~ax_mask_support),
                ax_n=np.maximum(self.min_bounds - ax_mask_support.sum(axis=1), 0)
            )
            ax_mask_support |= ax_sampled

        # Build support inputs
        sax_support_bits = (sax_active_inputs + self.original_inputs).multiply(
            self.bitmap.f2b(csc_matrix(ax_mask_support.T))
        )

        # Visualize result of shaper
        if self.advanced_plot_perf_enabled:
            self.visualize_multi_selection(sax_active_inputs, ax_mask_support)

        import IPython
        IPython.embed()
        return sax_support_bits

