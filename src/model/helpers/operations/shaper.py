# Global import
import numpy as np
from scipy.sparse import csc_matrix, diags
from scipy.signal import convolve2d

# Local import
from src.model.helpers.patterns import YalaBasePatterns
from src.model.helpers.operations.visualizer import Visualizer
from src.model.utils import sample_from_mask


class Shaper(Visualizer):
    """Shaper"""

    def __init__(
            self, server, sax_bf_map, drainer_params, min_firing=100, n_convex_bounds=3, perf_plotter=None,
    ):
        # call parent constructor
        self.n_convex_bounds = n_convex_bounds
        super().__init__(server, sax_bf_map, drainer_params, min_firing, perf_plotter=perf_plotter)

    def select_support_bits(self, sax_drained_weights, sax_count_activations):
        # Get active & drained bits
        sax_active_inputs = self.select_inputs(sax_drained_weights, sax_count_activations)
        sax_drained_inputs = sax_count_activations > 0

        # Compute input cardinality for each features
        ax_card_selected: np.ndarray = self.b2f(sax_active_inputs.astype(int)).A
        ax_card_signal: np.ndarray = self.b2f(sax_drained_inputs.astype(int)).A

        # Identify bounds to remove
        ax_mask_selected = ax_card_selected < ax_card_signal

        # Build support inputs
        sax_support_bits = (sax_active_inputs + self.pre_draining_fg.I).multiply(
            self.f2b(csc_matrix(ax_mask_selected.T))
        )
        # Add bounds used as mask
        sax_support_bits += self.fg_mask.I

        # Visualize result of shaper
        if self.advanced_plot_perf_enabled:
            self.visualize_multi_selection(sax_active_inputs, ax_mask_selected)

        return sax_support_bits

    def build_patterns(self, component, **d_signals):
        # Build convex hull
        sax_ch_inputs = self.build_random_convex_hull(component, d_signals['x'], d_signals['fg'])

        # Build components
        fg_comp, mask_comp = self.build_components(component, sax_ch_inputs)
        self.pre_draining_fg = YalaBasePatterns.from_fg_comp(fg_comp)

        # Build patterns
        self.fg_mask = YalaBasePatterns.from_fg_comp(mask_comp)
        self.firing_graph = YalaBasePatterns.from_fg_comp(fg_comp.copy(
            inputs=csc_matrix(fg_comp.inputs.A ^ self.f2b(self.b2f(fg_comp.inputs).T).A).multiply(sax_ch_inputs)
        ))

    def build_components(self, cmp, sax_ch, reduce_factor=0.4):

        # Init variables
        ax_reduced_inputs, sax_counts = np.zeros(sax_ch.shape, dtype=bool), self.b2f(sax_ch.astype(int))
        ax_lengths = np.unique(sax_counts.data)
        sax_counts.data = ((sax_counts.data * reduce_factor) / 2).round().clip(min=1).astype(int)

        # Reduce convex hull
        for w in ax_lengths:
            win_len = int(w + (w % 2 == 0))
            ax_win = np.ones((win_len, 1))
            ax_mask = self.bf_map.dot(sax_counts.A.T.astype(int))
            ax_reduced_inputs |= convolve2d(sax_ch.A, ax_win, mode='same') >= (win_len // 2) + 1 + ax_mask

        return cmp.copy(inputs=csc_matrix(ax_reduced_inputs), levels=np.ones(len(cmp))), cmp

    def build_random_convex_hull(self, component, sax_x, sax_fg):
        # Get masked activations
        sax_product = sax_x.astype(bool).T.dot(sax_fg)

        # Compute feature masks
        ax_card, ax_mask_component = self.b2f(sax_product.astype(int)).A, self.b2f(component.inputs.astype(bool)).A
        ax_card_mask = (0 < ax_card) & (ax_card < self.get_card_features(ax_card.shape[0]))

        # Select randomly n_convex_bounds that are not masked
        sax_sampled = csc_matrix(sample_from_mask(ax_card_mask & ~ax_mask_component, n=self.n_convex_bounds).T)

        return sax_product.multiply(self.f2b(sax_sampled))
