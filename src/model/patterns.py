# Global imports
import numpy as np
from scipy.sparse import lil_matrix

# Local import
from firing_graph.core.data_structure.graph import FiringGraph
from firing_graph.core.data_structure.utils import create_empty_matrices, add_core_vertices, reduce_matrices, \
    reduce_backward_firing, augment_matrices


class EmptyPattern(FiringGraph):

    def __init__(self, n_inputs, n_outputs, index_output, **kwargs):

        self.n_inputs, self.n_outputs, self.n_intersection, self.index_output = n_inputs, n_outputs, 0, index_output

        kwargs.update({
            'project': 'EmptyPattern', 'ax_levels': np.array([]), 'depth': 1,
            'matrices': create_empty_matrices(n_inputs, n_outputs, self.n_intersection)
        })

        # Invoke parent constructor
        super(EmptyPattern, self).__init__(**kwargs)


class YalaBasePattern(FiringGraph):
    """
    This class implement the main data structure used for fitting data. It is composed of weighted link in the form of
    scipy.sparse matrices and store complement information on vertices such as levels, mask for draining. It also keep
    track of the firing of vertices.

    """
    def __init__(self, n_inputs, n_outputs, index_output, **kwargs):

        self.n_inputs, self.n_outputs, self.index_output, self.n_intersection = n_inputs, n_outputs, index_output, 1
        kwargs.update({'project': 'YalaBasePattern', 'depth': 2})

        # Invoke parent constructor
        super(YalaBasePattern, self).__init__(**kwargs)

    @staticmethod
    def from_dict(d_struct, **graph_kwargs):

        return YalaBasePattern(
            n_inputs=d_struct['n_inputs'], n_outputs=d_struct['n_outputs'], index_output=d_struct['index_output'],
            **graph_kwargs
        )

    @staticmethod
    def from_partition(partition, firing_graph, index_output=None):

        # In case of empty partition, stop
        if len(partition['indices']) == 0:
            return None

        # Initialize matrices
        index_output = partition.get('index_output', index_output)
        d_matrices = reduce_matrices(firing_graph.matrices, partition['indices'])

        # Set levels and output link
        ax_levels = firing_graph.levels[partition['indices']]
        d_matrices['Ow'] = d_matrices['Ow'].tolil()
        d_matrices['Ow'][0, index_output] = 1

        # Add kwargs
        kwargs = {
            'partitions': partition.get('partitions', None), 'precision': partition.get('precision', None),
            'ax_levels': ax_levels, 'matrices': d_matrices
        }

        return YalaBasePattern(firing_graph.I.shape[0], firing_graph.O.shape[1], index_output, **kwargs)

    @staticmethod
    def from_input_indices(n_inputs, n_outputs, index_output, l_inputs, **kwargs):

        # Initialize Matrices
        d_matrices = create_empty_matrices(n_inputs, n_outputs, 1)

        # Set level and matrices
        ax_levels = np.array([len(l_inputs)])
        d_matrices['Iw'][l_inputs, 0] = 1
        d_matrices['Ow'][0, index_output] = 1

        # Add firing graph kwargs
        kwargs.update({'ax_levels': ax_levels, 'matrices': d_matrices})

        return YalaBasePattern(n_inputs, n_outputs, index_output, **kwargs)

    def augment(self, l_indices, precision=None):

        # Add inputs to intersection of interest
        self.matrices['Iw'] = self.matrices['Iw'].tolil()
        self.matrices['Iw'][l_indices, 0] = 1
        self.matrices['Iw'].tocsc()

        # Increase level accordingly
        self.levels += len(l_indices)

        if precision is not None:
            self.precision = precision

        return self

    def copy(self):

        d_graph = super(YalaBasePattern, self).to_dict(deep_copy=True)
        d_struct = {
            'n_inputs': self.n_inputs,
            'n_outputs': self.n_outputs,
            'index_output': self.index_output,
        }

        return self.from_dict(d_struct, **{k: v for k, v in d_graph.items()})


class YalaTransientPattern(FiringGraph):
    """
    This class implement the main data structure used for fiting data. It is composed of weighted link in the form of
    scipy.sparse matrices and store complement information on vertices such as levels, mask for draining. It also keep
    track of the firing of vertices.

    """
    def __init__(self, n_intersection, n_inputs, n_outputs, index_output, **kwargs):

        self.n_inputs, self.n_outputs, self.n_intersection = n_inputs, n_outputs, n_intersection
        self.index_output = index_output

        kwargs.update({'project': 'YalaTransientPattern', 'depth': 2 + int(self.n_intersection > 1)})

        # Invoke parent constructor
        super(YalaTransientPattern, self).__init__(**kwargs)

    @property
    def n_core(self):
        return self.n_intersection + int(self.n_intersection > 1)

    @staticmethod
    def from_dict(d_struct, **graph_kwargs):

        return YalaTransientPattern(
            n_intersection=d_struct['n_intersection'], n_inputs=d_struct['n_inputs'], n_outputs=d_struct['n_outputs'],
            index_output=d_struct['index_output'], **graph_kwargs
        )

    @staticmethod
    def from_partition(partition, firing_graph, index_output=None, add_backward_firing=False):

        # In case of empty partition, stop
        if len(partition['indices']) == 0:
            return None

        # Set number of intersection (remove top vertex in case of multiple intersection)
        n_inter = len(partition['indices']) - int(len(partition['indices']) > 1)

        # Set levels and latrices
        d_matrices = reduce_matrices(firing_graph.matrices, partition['indices'])
        ax_levels = firing_graph.levels[partition['indices']]
        d_matrices['Ow'] = d_matrices['Ow'].tolil()
        d_matrices['Ow'][-1, index_output] = 1

        # Add firing graph kwargs
        kwargs = {
            'partitions': partition.get('partitions', None), 'precision': partition.get('precision', None),
            'matrices': d_matrices, 'ax_levels': ax_levels
        }
        if add_backward_firing:
            kwargs.update(
                {'backward_firing': reduce_backward_firing(firing_graph.backward_firing, partition['indices'])}
            )

        return YalaTransientPattern(
            n_inter, firing_graph.I.shape[0], firing_graph.O.shape[1], partition.get('index_output', index_output),
            **kwargs
        )

    @staticmethod
    def from_input_indices(n_inputs, n_outputs, l0, index_output, l_inputs, weight, enable_drain=True, **kwargs):

        # Set levels
        n_intersection = len(l_inputs)
        ax_levels = np.array([l0] * n_intersection)

        if n_intersection > 1:
            ax_levels = np.hstack((ax_levels, np.array([1])))

        # Set Matrices
        d_matrices = create_empty_matrices(n_inputs, n_outputs, n_intersection + int(n_intersection > 1))
        for i, l_bits in enumerate(l_inputs):
            d_matrices['Iw'][list(l_bits), i] = weight

        for i in range(n_intersection - int(n_intersection == 1)):
            d_matrices['Cw'][i, n_intersection] = 1

        d_matrices['Ow'][n_intersection - int(n_intersection == 1), index_output] = 1

        # Update mask if necessary
        if enable_drain:
            d_matrices.update({'Im': d_matrices['Iw'] > 0})

        # Add firing graph kwargs
        kwargs.update({'ax_levels': ax_levels, 'matrices': d_matrices})

        return YalaTransientPattern(n_intersection, n_inputs, n_outputs, index_output, **kwargs)

    def copy(self):

        d_graph = super(YalaTransientPattern, self).to_dict(deep_copy=True)
        d_struct = {
            'n_inputs': self.n_inputs,
            'n_outputs': self.n_outputs,
            'n_intersection': self.n_intersection,
            'index_output': self.index_output,
        }
        return self.from_dict(d_struct, **{k: v for k, v in d_graph.items()})


class YalaAntiSymmetricBase(FiringGraph):
    """
    To do if needed: allow anti symmetric non activation.

    \/  \/
    O   O       levels = [l, l'] 
    | \/
    O O         levels = [1, 1]
    \/
    O           levels = [2]

    When left vertex of layer 1 activate alone then vertex of the last layer activate, otherwise it doesn't activate.
    """


class YalaSingleDrainingPattern(FiringGraph):
    """
    This class implement the main data structure used for fiting data. It is composed of weighted link in the form of
    scipy.sparse matrices and store complement information on vertices such as levels, mask for draining. It also keep
    track of the firing of vertices.

    """

    def __init__(self, n_inputs, n_outputs, index_output, **kwargs):

        self.n_inputs, self.n_outputs, self.index_output = n_inputs, n_outputs, index_output
        kwargs.update({'project': 'YalaSingleDrainingPattern'})

        # Invoke parent constructor
        super(YalaSingleDrainingPattern, self).__init__(**kwargs)

    @staticmethod
    def from_patterns(base_pattern, transient_pattern):

        depth, n_core = 3, base_pattern.n_intersection + transient_pattern.n_core
        matrices = augment_matrices(base_pattern.matrices, transient_pattern.matrices)
        ax_levels = np.hstack((base_pattern.levels, transient_pattern.levels))

        # If base structure is not empty add merging layer's vertices
        if base_pattern.n_intersection > 0:
            depth, n_core = depth + 1, n_core + 2
            matrices = add_core_vertices(matrices, 2, n_core - 2)
            ax_levels = np.hstack((ax_levels, np.array([1, 2])))

            # Link core vertices
            matrices['Cw'][base_pattern.n_intersection - 1, -2], matrices['Cw'][-2, -1] = 1, 1
            matrices['Cw'][base_pattern.n_intersection + transient_pattern.n_intersection, -1] = 1

        # Set output connection
        matrices['Ow'] = lil_matrix((n_core, base_pattern.n_outputs))
        matrices['Ow'][-1, base_pattern.index_output] = 1

        # Create partitions
        partitions = [
            {'indices': range(base_pattern.n_intersection), 'name': "base", 'precision': base_pattern.precision,
             "depth": 2},
            {'indices': [base_pattern.n_intersection + j for j in range(transient_pattern.n_intersection)],
             "name": "transient", "depth": 3}
        ]

        # Add firing graph kwargs
        kwargs = {
            'partitions': partitions, 'precision': base_pattern.precision, 'depth': depth, 'matrices': matrices,
            'ax_levels': ax_levels
        }

        return YalaSingleDrainingPattern(
            base_pattern.n_inputs, base_pattern.n_outputs, base_pattern.index_output, **kwargs
        )

    @staticmethod
    def from_partition(partition, firing_graph, index_output=None, add_backward_firing=False):

        ax_levels = firing_graph.levels[partition['indices']]
        matrices = reduce_matrices(firing_graph.matrices, partition['indices'])
        index_output = partition.get('index_output', index_output)

        # Add firing graph kwargs
        kwargs = {
            'partitions': partition.get('partitions', None), 'precision': partition.get('precision', None),
            'depth': partition['depth'], 'matrices': matrices, 'ax_levels': ax_levels
        }

        if add_backward_firing:
            kwargs.update(
                {'backward_firing': reduce_backward_firing(firing_graph.backward_firing, partition['indices'])}
            )

        return YalaSingleDrainingPattern(firing_graph.I.shape[0], firing_graph.O.shape[1], index_output, **kwargs)


class YalaMutlipleDrainingPattern(FiringGraph):

    def __init__(self, n_inputs, n_outputs, **kwargs):

        self.n_inputs, self.n_outputs = n_inputs, n_outputs
        kwargs.update({'project': 'YalaMutlipleDrainingPattern'})

        # Invoke parent constructor
        super(YalaMutlipleDrainingPattern, self).__init__(**kwargs)

    @staticmethod
    def check_patterns(l_patterns):
        assert all([isinstance(o, YalaSingleDrainingPattern) for o in l_patterns]),\
            "Only YalaSingleDrainingPattern can be used to build YalaMultipleDrainingPattern"

        assert len(set([o.depth for o in l_patterns])) == 1, \
            "Patterns of different depth inputed in YalaMultipleDrainingPattern"

    @staticmethod
    def from_patterns(l_patterns):

        if len(l_patterns) == 0:
            return None

        YalaMutlipleDrainingPattern.check_patterns(l_patterns)

        l_partitions, l_levels, depth = [], [], l_patterns[0].depth
        n_core_current, n_in, n_out = 0, l_patterns[0].n_inputs, l_patterns[0].n_outputs
        d_matrices = create_empty_matrices(n_in, n_out, 0)

        for pattern in l_patterns:

            # Set partitions
            l_partitions.append({
                'indices': [n_core_current + i for i in range(pattern.Cw.shape[1])],
                'depth': pattern.depth,
                'index_output': pattern.Ow.nonzero()[1][0]
            })

            if pattern.partitions is not None:
                l_partitions[-1].update({'partitions': pattern.partitions})

            if pattern.precision is not None:
                l_partitions[-1].update({'precision': pattern.precision})

            n_core_current += pattern.Cw.shape[1]
            d_matrices = augment_matrices(d_matrices, pattern.matrices, write_mode=None)

            # Merge levels
            l_levels.extend(list(pattern.levels))

        # Add firing graph kwargs
        kwargs = {
            'partitions': l_partitions, 'ax_levels': np.array(l_levels), 'matrices': d_matrices, 'depth': depth,
        }

        return YalaMutlipleDrainingPattern(n_in, n_out, **kwargs)


class YalaPredictingPattern(FiringGraph):
    """
    This class implement the main data structure used for fitting data. It is composed of weighted link in the form of
    scipy.sparse matrices and store complement information on vertices such as levels, mask for draining. It also keep
    track of the firing of vertices.

    """

    def __init__(self, n_inputs, n_outputs, **kwargs):

        self.n_inputs, self.n_outputs = n_inputs, n_outputs
        kwargs.update({'project': 'YalaDrainingPattern', 'depth': 2})

        # Invoke parent constructor
        super(YalaPredictingPattern, self).__init__(**kwargs)

    @staticmethod
    def check_patterns(l_patterns):
        assert all([isinstance(o, YalaBasePattern) for o in l_patterns]),\
            "Only YalaPredictingPattern can be used to build YalaBasePattern"

        assert len(set([o.depth for o in l_patterns])) == 1, \
            "Patterns of different depth inputed in YalaPredictingPattern"

    @staticmethod
    def from_base_patterns(l_base_patterns):

        if len(l_base_patterns) == 0:
            return None

        # check patterns and intitialize variables
        YalaPredictingPattern.check_patterns(l_base_patterns)
        l_partitions, n_core, l_levels = [], 0, []
        n_inputs, n_outputs = l_base_patterns[0].n_inputs, l_base_patterns[0].n_outputs
        d_matrices = create_empty_matrices(n_inputs, n_outputs, 0)

        # Build pattern from list of patterns
        for pattern in l_base_patterns:

            # Set partitions
            l_partitions.append({
                'indices': [n_core],
                'depth': pattern.depth,
                'index_output': pattern.index_output,
                'precision': pattern.precision
            })

            # Augment matrices and levels
            d_matrices = augment_matrices(d_matrices, pattern.matrices, write_mode=None)
            l_levels.extend(list(pattern.levels))
            n_core += 1

        # Add firing graph kwargs
        kwargs = {'partitions': l_partitions, 'matrices': d_matrices, 'ax_levels': np.array(l_levels)}

        return YalaPredictingPattern(l_base_patterns[0].n_inputs, l_base_patterns[0].n_outputs, **kwargs)

    def augment(self, l_base_patterns):

        if len(l_base_patterns) == 0:
            return self

        self.check_patterns(l_base_patterns)

        # merge partitions
        n_core = self.C.shape[0]
        for pattern in l_base_patterns:

            # Set partitions
            self.partitions.append({
                'indices': [n_core],
                'depth': pattern.depth,
                'index_output': pattern.index_output,
                'precision': pattern.precision
            })

            # Augment matrices and levels
            self.matrices = augment_matrices(self.matrices, pattern.matrices, write_mode=False)
            self.levels = np.hstack((self.levels, pattern.levels))
            n_core += 1

        return self
