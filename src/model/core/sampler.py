# Global import
import numpy as np
from random import choices
from string import ascii_uppercase
from scipy.signal import convolve2d
from scipy.sparse import csc_matrix

# Local import
from src.model.utils.data_models import ConvexHullProba, FgComponents
from src.model.utils.firing_graph import YalaFiringGraph


class Sampler:
    """Sampling bounds for firing graph"""

    def __init__(self, server, bitmap, n_bounds=2):
        self.server = server
        self.bitmap = bitmap
        self.n_bounds = n_bounds
        self.ch_probas = ConvexHullProba()

    def init_sample(self, n_verts, window_length):
        # Get dimensions
        (n_inputs, n_features) = self.bitmap.bf_map.shape

        # Sample features
        ax_indices = np.random.choice(n_features, self.n_bounds * n_verts)
        ax_mask = np.zeros((n_features, n_verts), dtype=bool)
        ax_mask[ax_indices, np.array([i // self.n_bounds for i in range(self.n_bounds * n_verts)])] = True

        # Sample inputs and expand it
        ax_sampled = self.server.get_random_samples(n_verts).A.astype(int).T * \
            self.bitmap.bf_map.A.dot(ax_mask.astype(int))

        # Use convolution to explore neighbour area and re-mask
        ax_inputs = convolve2d(ax_sampled, np.ones((window_length, 1)), mode='same')
        ax_inputs = ax_inputs * self.bitmap.bf_map.A.dot(ax_mask.astype(int))

        # Create comp and compute precisions
        comp = FgComponents(
            inputs=csc_matrix(ax_inputs), levels=(ax_inputs.T.dot(self.bitmap.bf_map.A) > 0).sum(axis=1),
            partitions=[
                {'label_id': 0, 'id': ''.join(choices(ascii_uppercase, k=5)), "stage": "ongoing"}
                for _ in range(n_verts)
            ],
        )

        return comp

    def sample_bounds(self, base_components, batch_size):
        # Get convex components
        ch_components = YalaFiringGraph.from_fg_comp(base_components) \
            .get_convex_hull(self.server, batch_size)

        # Sample new bounds from CH
        sax_sampled = csc_matrix(self.sample_from_proba(base_components).T)

        # Update base component's bounds TODO: check dims matches
        base_components = base_components.update(
            inputs=base_components.inputs + ch_components.inputs.multiply(self.bitmap.f2b(sax_sampled)),
            levels=base_components.levels + sax_sampled.sum(axis=1)
        )

        # Update bounds proba
        self.ch_probas.add(ch_components, self.bitmap)

        return base_components

    def sample_from_proba(self, comps):

        ax_p = self.ch_probas.get_probas(comps, self.bitmap)

        # Prepare choice
        (ny, nx), ax_linear = ax_p.shape, np.arange(ax_p.shape[1])
        ax_counts = np.minimum((ax_p > 0).sum(axis=1), self.n_bounds) * ((ax_p > 0).sum(axis=1) >= self.n_bounds)

        def masked_choice(ax_p_, n_):
            return list(np.random.choice(ax_linear, n_, replace=False, p=ax_p_))

        # Random coordinate choice
        l_y_ind = sum([[i] * ax_counts[i] for i in range(ny)], [])
        l_x_ind = sum([masked_choice(ax_p[i, :], ax_counts[i]) for i in range(ny)], [])

        # Create new mask features
        ax_mask_sampled = np.zeros((ny, nx), dtype=bool)
        if l_y_ind:
            ax_mask_sampled[l_y_ind, l_x_ind] = True

        return ax_mask_sampled
