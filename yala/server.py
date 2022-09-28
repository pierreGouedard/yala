# Global import
import numpy as np
from random import choices
from string import ascii_uppercase
from scipy.sparse import csr_matrix, lil_matrix

# Local import
from yala.utils.data_models import ConvexHullProba, FgComponents
from .firing_graph import YalaFiringGraph
from yala.linalg.spmat_op import expand


class Sampler:
    """Sampling bounds for firing graph"""

    def __init__(self, server, bitmap, n_bounds=2):
        self.server = server
        self.bitmap = bitmap
        self.n_bounds = n_bounds
        self.ch_probas = ConvexHullProba()

    def init_sample(self, n_verts, n_bits):
        # Get dimensions
        (n_inputs, n_features) = self.bitmap.bf_map.shape

        # Sample features
        # TODO: tmp test
        sax_sampled = expand(
            self.server.get_random_samples(n_verts).T, self.bitmap, n_bits // 2
        )
        #
        #ax_indices = np.hstack([np.random.choice(n_features, self.n_bounds, replace=False) for _ in range(n_verts)])
        #sax_mask = lil_matrix((n_features, n_verts), dtype=bool)
        #sax_mask[ax_indices, np.array([i // self.n_bounds for i in range(self.n_bounds * n_verts)])] = True

        # Sample inputs and expand it
        #sax_sampled = expand(
        #    self.server.get_random_samples(n_verts).T.multiply(self.bitmap.f2b(sax_mask.tocsr())),
        #    self.bitmap, n_bits // 2
        #)
        # Create comp and compute precisions
        comp = FgComponents(
            inputs=sax_sampled, mask_inputs=sax_sampled, levels=self.bitmap.b2f(sax_sampled).A.sum(axis=1),
            partitions=[
                {'label_id': 0, 'id': ''.join(choices(ascii_uppercase, k=5)), "stage": "ongoing"}
                for _ in range(sax_sampled.shape[1])
            ],
        )

        return comp

    def sample_bounds(self, base_components, batch_size, n_bounds):
        # Get convex components
        ch_components = YalaFiringGraph.from_comp(base_components) \
            .get_convex_hull(self.server, batch_size, self.bitmap)

        # Sample new bounds from CH
        sax_sampled = csr_matrix(self.sample_from_proba(base_components, n_bounds).T, dtype=bool)
        ch_components.update(inputs=ch_components.inputs.multiply(self.bitmap.f2b(sax_sampled)))

        # Update base component's bounds
        base_components = base_components.update(
            inputs=base_components.inputs + ch_components.inputs,
            levels=base_components.levels + sax_sampled.sum(axis=0).A[0, :]
        )

        # Update bounds proba
        self.ch_probas.add(ch_components, self.bitmap)

        return base_components

    def sample_from_proba(self, comps, n_bounds=None):
        ax_p = self.ch_probas.get_probas(comps, self.bitmap)
        n_bounds = n_bounds or self.n_bounds

        # Prepare choice
        (ny, nx), ax_linear = ax_p.shape, np.arange(ax_p.shape[1])
        ax_counts = np.minimum((ax_p > 0).sum(axis=1), n_bounds) * ((ax_p > 0).sum(axis=1) >= n_bounds)

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
