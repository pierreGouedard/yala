# Global import
from random import choices
from string import ascii_uppercase
from numpy import arange, zeros
from numpy.random import choice
from scipy.sparse import csr_matrix

# Local import
from firing_graph.servers import ArrayServer
from yala.firing_graph import YalaFiringGraph
from yala.utils.data_models import FgComponents, BitMap
from yala.linalg.spmat_op import expand


class YalaServer(ArrayServer):
    """Specific server for YALA app"""

    def __init__(self, X, y, bf_map, n_bounds_start=2, n_bounds_incr=2):
        self.__bf_map = bf_map
        self.__sax_forward = X.tocsr()

        # Get first feature mask and update data forward
        self.feature_mask, self.bit_mask = self.init_masks(bf_map, bf_map.shape[1], n_bounds_start)
        self.sax_forward = self.__sax_forward.multiply(csr_matrix(self.bit_mask))

        # Bild current bitmask
        self.curr_bitmap = BitMap(bf_map[:, self.feature_mask], bf_map.shape[0], self.feature_mask.sum())

        # Other params
        self.n_bounds_start = n_bounds_start
        self.n_bounds_incr = n_bounds_incr

        super().__init__(X, y)

    @property
    def bitmap(self):
        return self.curr_bitmap

    @staticmethod
    def init_masks(bf_map, n_features, n_bounds_start):
        ax_feature_mask = zeros(n_features, dtype=bool)
        ax_feature_mask[arange(0, n_features, n_features // n_bounds_start)] = True
        ax_bit_mask = bf_map[:, ax_feature_mask].A.any(axis=1)

        return ax_feature_mask, ax_bit_mask

    def update_masks(self, ):
        # Get remaining candidate
        l_candidates = [i for i in range(len(self.feature_mask)) if not self.feature_mask[i]]
        if not len(l_candidates):
            return l_candidates

        l_new = choice(l_candidates, self.n_bounds_incr, replace=False)

        # Update feature and bitmask
        self.feature_mask[l_new] = True
        self.bit_mask = self.__bf_map[:, self.feature_mask].A.any(axis=1)
        self.sax_forward = self.__sax_forward.multiply(csr_matrix(self.bit_mask))
        self.curr_bitmap = BitMap(self.__bf_map[:, self.feature_mask], self.__bf_map.shape[0], self.feature_mask.sum())

        return l_new

    def init_sampling(self, n_verts, n_bits):
        # Sample data point
        sax_sampled = expand(
            self.get_random_samples(n_verts).multiply(csr_matrix(self.bit_mask, dtype=bool)).T,
            self.bitmap, n_bits // 2
        ).multiply(csr_matrix(self.bit_mask, dtype=bool).T)

        # Create comp and compute precisions
        comps = FgComponents(
            inputs=sax_sampled, mask_inputs=sax_sampled, levels=self.bitmap.b2f(sax_sampled).A.sum(axis=1),
            partitions=[
                {'label_id': 0, 'id': ''.join(choices(ascii_uppercase, k=5)), "stage": "ongoing"}
                for _ in range(sax_sampled.shape[1])
            ],
        )

        return comps

    def update_bounds(self, comps):

        l_new_indices = self.update_masks()
        if not len(l_new_indices) or comps.empty:
            return True

        ch_comps = YalaFiringGraph.from_comp(comps).get_convex_hull(self)
        ch_comps.update(
            inputs=ch_comps.inputs.multiply(sum([self.__bf_map[:, int(i)] for i in l_new_indices]))
        )
        comps.inputs = comps.inputs + ch_comps.inputs
        comps.levels = self.bitmap.b2f(comps.inputs).sum(axis=1).A[:, 0]

        return False
