# Global imports
import numpy as np
from itertools import groupby
from scipy.sparse import eye, csr_matrix

# Local import
from firing_graph.graph import FiringGraph
from firing_graph.graph import create_empty_matrices
from yala.utils.data_models import FgComponents
from yala.linalg.spmat_op import fill_gap


class YalaFiringGraph(FiringGraph):
    """
    This class implement the main data structure used for fitting data. It is composed of weighted link in the form of
    scipy.sparse matrices and store complement information on vertices such as levels, mask for draining. It also keep
    track of the firing of vertices.

    """

    def __init__(self, **kwargs):

        kwargs.update({'project': 'YalaFiringGraph', 'depth': 2})

        # Invoke parent constructor
        super(YalaFiringGraph, self).__init__(**kwargs)

    def to_comp(self):
        return FgComponents(inputs=self.I, mask_inputs=self.Im, levels=self.levels, partitions=self.partitions)

    @staticmethod
    def from_comp(fg_comp: FgComponents):

        if len(fg_comp) == 0:
            return None

        # Get output indices and initialize matrices
        d_matrices = create_empty_matrices(
            fg_comp.inputs.shape[0], fg_comp.inputs.shape[1], fg_comp.inputs.shape[1], write_mode=False
        )

        # Set matrices
        d_matrices['Iw'] = fg_comp.inputs.astype(np.int32)
        d_matrices['Im'] = fg_comp.mask_inputs
        d_matrices['O'] += eye(fg_comp.inputs.shape[1], format='csr', dtype=bool)

        # Add firing graph kwargs
        kwargs = {
            'partitions': fg_comp.partitions, 'matrices': d_matrices, 'ax_levels': fg_comp.levels,
        }

        return YalaFiringGraph(**kwargs)

    @staticmethod
    def from_inputs(sax_inputs, sax_mask_inputs, levels, partitions):

        # Get output indices and initialize matrices
        d_matrices = create_empty_matrices(
            sax_inputs.shape[0], sax_inputs.shape[1], sax_inputs.shape[1], write_mode=False
        )

        # Set matrices
        d_matrices['Iw'] = sax_inputs.astype(np.int32)
        d_matrices['Im'] = sax_mask_inputs
        d_matrices['O'] += eye(sax_mask_inputs.shape[1], format='csr', dtype=bool)

        # Add firing graph kwargs
        kwargs = {
            'partitions': partitions, 'matrices': d_matrices, 'ax_levels': levels,
        }

        return YalaFiringGraph(**kwargs)

    def get_convex_hull(self, server):
        # Get masked activations
        sax_x = server.next_all_forward().sax_data_forward

        # propagate through firing graph
        sax_fg = self.seq_propagate(sax_x)

        # Get masked activations
        sax_product = sax_x.T.dot(sax_fg)

        return FgComponents(
            inputs=fill_gap(sax_product, server.bitmap), mask_inputs=csr_matrix((0, 0)),
            levels=np.ones(sax_fg.shape[1]), partitions=self.partitions
        )


class YalaTopPattern(FiringGraph):
    """
    This class implement the main data structure used for fitting data. It is composed of weighted link in the form of
    scipy.sparse matrices and store complement information on vertices such as levels, mask for draining. It also keep
    track of the firing of vertices.

    """
    def __init__(self, n_inputs, l_labels):
        # Build sparse matrices
        self.n_inputs, self.n_outputs = n_inputs, len(l_labels)
        d_matrices, l_partitions = self.build(l_labels, self.n_inputs, self.n_outputs)

        # Build kwargs
        kwargs = {
            'partitions': l_partitions, 'matrices': d_matrices, 'project': 'YalaTopPattern', 'depth': 2,
            'ax_levels': np.ones(self.n_outputs)
        }

        # Invoke parent constructor
        super(YalaTopPattern, self).__init__(**kwargs)

    @staticmethod
    def build(l_labels, n_inputs, n_outputs):

        d_matrices, l_partitions = create_empty_matrices(n_inputs, n_outputs, n_outputs), []
        d_matrices['O'] += eye(n_outputs, format='csc', dtype=int)

        # Build I and partitions
        gr = groupby([(l, i) for i, l in enumerate(l_labels)], key=lambda t: t[0])
        for k, v in gr:
            l_inds = list(map(lambda x: x[1], v))
            d_matrices['Iw'][int(k), l_inds] = 1
            l_partitions.append({'indices': l_inds, 'index_input': int(k)})

        return d_matrices, l_partitions
