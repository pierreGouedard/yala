# Global imports
import numpy as np
from itertools import groupby
from scipy.sparse import lil_matrix, hstack, eye

# Local import
from firing_graph.data_structure.graph import FiringGraph
from firing_graph.data_structure.utils import create_empty_matrices, set_matrices_spec
from src.model.core.data_models import FgComponents


class YalaBasePatterns(FiringGraph):
    """
    This class implement the main data structure used for fitting data. It is composed of weighted link in the form of
    scipy.sparse matrices and store complement information on vertices such as levels, mask for draining. It also keep
    track of the firing of vertices.

    """

    def __init__(self, n_vertex, **kwargs):

        self.n_vertex = n_vertex
        kwargs.update({'project': 'YalaBasePatterns', 'depth': 2})

        # Invoke parent constructor
        super(YalaBasePatterns, self).__init__(**kwargs)

    @staticmethod
    def from_dict(d_struct, **graph_kwargs):
        return YalaBasePatterns(n_vertex=d_struct['n_vertex'], **graph_kwargs)

    @staticmethod
    def from_fg_comp(fg_comp):

        if len(fg_comp) == 0:
            return None

        # Get output indices and initialize matrices
        d_matrices = create_empty_matrices(
            n_inputs=fg_comp.inputs.shape[0], n_outputs=fg_comp.inputs.shape[1], n_core=fg_comp.inputs.shape[1]
        )

        # Set matrices
        d_matrices['Iw'] = fg_comp.inputs.copy()
        d_matrices['Ow'] += eye(fg_comp.inputs.shape[1], format='csc', dtype=int)

        # Add firing graph kwargs
        kwargs = {'partitions': fg_comp.partitions, 'matrices': d_matrices, 'ax_levels': fg_comp.levels}

        return YalaBasePatterns(d_matrices['Iw'].shape[1], **kwargs)

    def augment_from_fg_comp(self, fg_comp):
        if len(fg_comp) == 0:
            return self

        return YalaBasePatterns.from_fg_comp(FgComponents(
            inputs=hstack([self.I, fg_comp.inputs]), partitions=self.partitions + fg_comp.partitions,
            levels=np.hstack((self.levels, fg_comp.levels))
        ))

    def reduce_output(self):

        # Get labels
        l_labels = [p['label_id'] for p in self.partitions]

        # Map vertices to label output
        sax_reducer = lil_matrix((self.n_vertex, max(l_labels) + 1))
        sax_reducer[np.arange(self.n_vertex), l_labels] = 1
        self.matrices['Ow'] = self.matrices['Ow'].dot(sax_reducer)

        # reformat matrices
        set_matrices_spec(self.matrices, write_mode=False)

        return self

    def copy(self):

        d_graph = super(YalaBasePatterns, self).to_dict(deep_copy=True)
        d_struct = {
            'n_vertex': self.n_vertex,
        }
        return self.from_dict(d_struct, **{k: v for k, v in d_graph.items()})


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
        d_matrices['Ow'] += eye(n_outputs, format='csc', dtype=int)

        # Build I and partitions
        gr = groupby([(l, i) for i, l in enumerate(l_labels)], key=lambda t: t[0])
        for k, v in gr:
            l_inds = list(map(lambda x: x[1], v))
            d_matrices['Iw'][int(k), l_inds] = 1
            l_partitions.append({'indices': l_inds, 'index_input': int(k)})

        return d_matrices, l_partitions