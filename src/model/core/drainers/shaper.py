# Global import
import numpy as np
from scipy.sparse import csc_matrix, diags
from scipy.signal import convolve2d

# Local import
from src.model.core.patterns import YalaBasePatterns
from src.model.core.drainers.visualizer import Visualizer
from src.model.utils import sample_from_proba


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

        # Identify bounds to select
        ax_mask_selected = ax_card_selected < ax_card_signal

        # Build support inputs
        sax_support_bits = (sax_active_inputs + self.pre_draining_inputs).multiply(
            self.f2b(csc_matrix(ax_mask_selected.T))
        )

        # Mask's bound that have not been selected
        sax_mask_mask = self.f2b(csc_matrix(self.b2f(self.fg_mask.I).A ^ ax_mask_selected, dtype=bool).T)
        sax_support_bits += self.fg_mask.I.multiply(sax_mask_mask)

        # Visualize result of shaper
        if self.advanced_plot_perf_enabled:
            self.visualize_multi_selection(sax_active_inputs, ax_mask_selected)
        import IPython
        IPython.embed()
        return sax_support_bits

    def build_patterns(self, component, **kwargs):
        # Build convex hull
        sax_ch_inputs = self.build_random_hull(component, kwargs['x'], kwargs['fg'], kwargs['shaper_probas'])

        # Build components
        fg_comp, mask_comp = self.build_components(component, sax_ch_inputs)
        self.pre_draining_inputs = fg_comp.inputs.copy()

        # Build patterns
        self.fg_mask = YalaBasePatterns.from_fg_comp(mask_comp)
        self.firing_graph = YalaBasePatterns.from_fg_comp(fg_comp.copy(
            inputs=csc_matrix((fg_comp.inputs.A ^ sax_ch_inputs.A))
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

    def build_random_hull(self, component, sax_x, sax_fg, shaper_probas):
        # Get masked activations
        sax_product = sax_x.astype(bool).T.dot(sax_fg)

        # Compute feature masks
        ax_card, ax_mask_component = self.b2f(sax_product.astype(int)).A, self.b2f(component.inputs.astype(bool)).A
        ax_card_mask = (0 < ax_card) & (ax_card < self.get_card_features(ax_card.shape[0]))

        # Get proba
        ax_p = shaper_probas.probas.T * ax_card_mask

        # Select randomly n_convex_bounds that are not masked
        sax_sampled = csc_matrix(sample_from_proba(ax_p / ax_p.sum(axis=1), n=self.n_convex_bounds).T)

        return sax_product.multiply(self.f2b(sax_sampled))
