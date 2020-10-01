# Global imports
import numpy as np
from scipy.sparse import lil_matrix, eye, diags

# Local import
from firing_graph.core.data_structure.graph import FiringGraph
from firing_graph.core.data_structure.utils import create_empty_matrices, add_core_vertices, reduce_matrices, \
    reduce_backward_firing, augment_matrices


class EmptyPattern(FiringGraph):

    def __init__(self, n_inputs, n_outputs, label_id, output_id, **kwargs):

        self.n_inputs, self.n_outputs, self.size = n_inputs, n_outputs, 0
        self.label_id, self.output_id = label_id, output_id
        kwargs.update({
            'project': 'EmptyPattern', 'ax_levels': np.array([]), 'depth': 0,
            'matrices': create_empty_matrices(n_inputs, n_outputs, self.size)
        })

        # Invoke parent constructor
        super(EmptyPattern, self).__init__(**kwargs)


class YalaBasePattern(FiringGraph):
    """
    This class implement the main data structure used for fitting data. It is composed of weighted link in the form of
    scipy.sparse matrices and store complement information on vertices such as levels, mask for draining. It also keep
    track of the firing of vertices.

    """
    def __init__(self, n_inputs, n_outputs, label_id, output_id, **kwargs):

        self.n_inputs, self.n_outputs, self.size = n_inputs, n_outputs, 1
        self.label_id, self.output_id = label_id, output_id,

        kwargs.update({'project': 'YalaBasePattern', 'depth': 2})

        # Invoke parent constructor
        super(YalaBasePattern, self).__init__(**kwargs)

    @staticmethod
    def from_dict(d_struct, **graph_kwargs):

        return YalaBasePattern(
            n_inputs=d_struct['n_inputs'], n_outputs=d_struct['n_outputs'], label_id=d_struct['label_id'],
            output_id=d_struct['output_id'], **graph_kwargs
        )

    @staticmethod
    def from_partition(partition, firing_graph, label_id=None, output_id=None, add_backward_firing=False):

        # In case of empty partition, stop
        if len(partition['indices']) == 0:
            return None

        # Initialize matrices
        label_id, output_id = partition.get('label_id', label_id), partition.get('output_id', output_id)
        d_matrices = reduce_matrices(firing_graph.matrices, partition['indices'])

        # Set levels and output link
        ax_levels = firing_graph.levels[partition['indices']]
        d_matrices['Ow'] = d_matrices['Ow'].tolil()
        d_matrices['Ow'][0, output_id] = 1

        # Add kwargs
        kwargs = {
            'partitions': partition.get('partitions', None), 'precision': partition.get('precision', None),
            'ax_levels': ax_levels, 'matrices': d_matrices, 'score': partition.get('score', None)
        }

        if add_backward_firing:
            kwargs.update(
                {'backward_firing': reduce_backward_firing(firing_graph.backward_firing, partition['indices'])}
            )

        return YalaBasePattern(firing_graph.I.shape[0], firing_graph.O.shape[1], label_id, output_id, **kwargs)

    @staticmethod
    def from_input_indices(n_inputs, n_outputs, label_id, output_id, l_inputs, l0, weight, enable_drain=True, **kwargs):

        # Initialize Matrices
        d_matrices = create_empty_matrices(n_inputs, n_outputs, 1)

        # Set level and matrices
        ax_levels = np.array([l0])
        d_matrices['Iw'][l_inputs, 0] = weight
        d_matrices['Ow'][0, output_id] = 1

        # Update mask if necessary
        if enable_drain:
            d_matrices.update({'Im': d_matrices['Iw'] > 0})

        # Add firing graph kwargs
        kwargs.update({'ax_levels': ax_levels, 'matrices': d_matrices})

        return YalaBasePattern(n_inputs, n_outputs, label_id, output_id, **kwargs)

    def update_outputs(self, output_id, n_outputs=None):

        # Get new n_outputs if specified
        self.n_outputs = n_outputs if n_outputs is not None else self.n_outputs

        # Change Output matrices
        sax_Ow = lil_matrix((1, self.n_outputs))
        sax_Ow[0, output_id] = 1
        self.matrices['Ow'] = sax_Ow.tocsc()

        # Change output mask
        sax_Om = lil_matrix((1, self.n_outputs))
        sax_Om[0, output_id] = self.matrices['Om'][0, self.output_id]
        self.matrices['Om'] = sax_Om.tocsc()

        # Change index output
        self.output_id = output_id

        return self

    def augment(self, l_indices, level_increment=0, precision=None):

        # Add inputs to intersection of interest
        self.matrices['Iw'] = self.matrices['Iw'].tolil()
        self.matrices['Iw'][l_indices, 0] = 1
        self.matrices['Iw'].tocsc()

        # Increase level accordingly
        self.levels += level_increment

        if precision is not None:
            self.precision = precision

        return self

    def to_dict(self, **kwargs):
        d_graph = super(YalaBasePattern, self).to_dict(deep_copy=True)
        d_struct = {'n_inputs': self.n_inputs, 'n_outputs': self.n_outputs, 'label_id': self.label_id,
                    'output_id': self.output_id}
        return d_struct, d_graph

    def copy(self):

        d_graph = super(YalaBasePattern, self).to_dict(deep_copy=True)
        d_struct = {
            'n_inputs': self.n_inputs,
            'n_outputs': self.n_outputs,
            'label_id': self.label_id,
            'output_id': self.output_id,
        }

        return self.from_dict(d_struct, **{k: v for k, v in d_graph.items()})


class YalaBasePatterns(FiringGraph):
    """
    This class implement the main data structure used for fitting data. It is composed of weighted link in the form of
    scipy.sparse matrices and store complement information on vertices such as levels, mask for draining. It also keep
    track of the firing of vertices.

    """

    def __init__(self, n_inputs, n_outputs, **kwargs):

        self.n_inputs, self.n_outputs = n_inputs, n_outputs
        kwargs.update({'project': 'YalaBasePatterns', 'depth': 2})

        # Invoke parent constructor
        super(YalaBasePatterns, self).__init__(**kwargs)

    @staticmethod
    def from_dict(d_struct, **graph_kwargs):
        return YalaBasePatterns(n_inputs=d_struct['n_inputs'], n_outputs=d_struct['n_outputs'], **graph_kwargs)

    @staticmethod
    def check_patterns(l_patterns):
        assert all([isinstance(o, YalaBasePattern) for o in l_patterns]),\
            "Only YalaBasePattern can be used to build YalaBasePatterns"

        assert len(set([o.depth for o in l_patterns])) == 1, \
            "Patterns of different depth inputed in YalaBasePatterns"

    @staticmethod
    def from_patterns(l_base_patterns, group_id=0, keep_output_id=False):

        if len(l_base_patterns) == 0:
            return None

        # check patterns and intitialize variables
        YalaBasePatterns.check_patterns(l_base_patterns)
        l_partitions, n_core, l_levels = [], 0, []
        n_inputs, n_outputs = l_base_patterns[0].n_inputs, l_base_patterns[0].n_outputs
        d_matrices = create_empty_matrices(n_inputs, n_outputs, 0)

        # Build pattern from list of patterns
        for pattern in l_base_patterns:

            # Set partitions
            l_partitions.append({
                'indices': [n_core],
                'depth': pattern.depth,
                'precision': pattern.precision,
                'score': pattern.score,
                'label_id': pattern.label_id,
                'group_id': group_id + pattern.label_id,
                'output_id': pattern.output_id if keep_output_id else pattern.label_id
            })

            # Augment matrices and levels
            d_matrices = augment_matrices(d_matrices, pattern.matrices, write_mode=None)
            l_levels.extend(list(pattern.levels))
            n_core += 1

        # Add firing graph kwargs
        kwargs = {'partitions': l_partitions, 'matrices': d_matrices, 'ax_levels': np.array(l_levels)}

        return YalaBasePatterns(l_base_patterns[0].n_inputs, l_base_patterns[0].n_outputs, **kwargs)

    @staticmethod
    def from_input_matrix(sax_I, l_partitions, ax_levels):

        assert sax_I.shape[1] == len(l_partitions), "The # core vertices is not equal to the length of partitions"

        if len(l_partitions) == 0:
            return None

        # Get output indices and intialize matrices
        t_outs = tuple(d_pred['output_id'] for d_pred in l_partitions)
        d_matrices = create_empty_matrices(
            n_inputs=sax_I.shape[0], n_outputs=max(t_outs) + 1, n_core=sax_I.shape[1]
        )

        # Set matrices
        d_matrices['Iw'] = sax_I
        d_matrices['Ow'][range(sax_I.shape[1]), t_outs] = 1

        # Add firing graph kwargs
        kwargs = {'partitions': l_partitions, 'matrices': d_matrices, 'ax_levels': ax_levels}

        return YalaBasePatterns(d_matrices['Iw'].shape[0], d_matrices['Ow'].shape[1], **kwargs)

    @staticmethod
    def from_partitions(l_partitions, firing_graph):
        # Extract core vertices input and level of interest
        sax_I = firing_graph.I[:, sum([p['indices'] for p in l_partitions], [])]
        ax_levels = firing_graph.levels[sum([p['indices'] for p in l_partitions], [])]

        # Update partitions
        l_partitions = [{k: p[k] if k != 'indices' else i for k in p.keys()} for i, p in enumerate(l_partitions)]

        return YalaBasePatterns.from_input_matrix(sax_I, l_partitions, ax_levels)

    def augment(self, l_base_patterns, group_id, keep_output_id=False):

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
                'precision': pattern.precision,
                'score': pattern.score,
                'label_id': pattern.label_id,
                'group_id': group_id + pattern.label_id,
                'output_id': pattern.output_id if keep_output_id else pattern.label_id
            })

            # Augment matrices and levels
            self.matrices = augment_matrices(self.matrices, pattern.matrices, write_mode=False)
            self.levels = np.hstack((self.levels, pattern.levels))
            n_core += 1

        return self

    def group_output(self):
        n_outputs = max([p['group_id'] for p in self.partitions]) + 1

        # Init output matrix
        sax_Ow = lil_matrix((self.Ow.shape[0], n_outputs))
        sax_Om = lil_matrix((self.Om.shape[0], n_outputs))

        for p in self.partitions:
            p['output_id'] = p['group_id']
            sax_Ow[p['indices'], p['group_id']] = True
            sax_Om[p['indices'], p['group_id']] = True

        self.matrices['Ow'] = sax_Ow.tocsc()
        self.matrices['Om'] = sax_Om.tocsc()

        return self

    def ungroup_output(self):
        n_outputs = max([p['label_id'] for p in self.partitions]) + 1

        # Init output matrix
        sax_Ow = lil_matrix((self.Ow.shape[0], n_outputs))
        sax_Om = lil_matrix((self.Om.shape[0], n_outputs))

        for p in self.partitions:
            p['output_id'] = p['label_id']
            sax_Ow[p['indices'], p['label_id']] = True
            sax_Om[p['indices'], p['label_id']] = True

        self.matrices['Ow'] = sax_Ow.tocsc()
        self.matrices['Om'] = sax_Om.tocsc()

        return self

    def isolate_output(self):
        # Set group_id
        for p in self.partitions:
            p['group_id'] = p['output_id']

        return self.group_output()

    def copy(self):

        d_graph = super(YalaBasePatterns, self).to_dict(deep_copy=True)
        d_struct = {
            'n_inputs': self.n_inputs,
            'n_outputs': self.n_outputs,
        }

        return self.from_dict(d_struct, **{k: v for k, v in d_graph.items()})


class YalaDrainingPattern(FiringGraph):
    """
    This class implement the main data structure used for fiting data. It is composed of weighted link in the form of
    scipy.sparse matrices and store complement information on vertices such as levels, mask for draining. It also keep
    track of the firing of vertices.

    """

    def __init__(self, n_inputs, n_outputs, label_id, output_id, **kwargs):

        self.n_inputs, self.n_outputs, self.label_id, self.output_id = n_inputs, n_outputs, label_id, output_id
        kwargs.update({'project': 'YalaDrainingPattern'})

        # Invoke parent constructor
        super(YalaDrainingPattern, self).__init__(**kwargs)

    @staticmethod
    def from_patterns(pred_pattern, transient_pattern):

        depth, n_core = 2, pred_pattern.size + transient_pattern.size,
        matrices = augment_matrices(pred_pattern.matrices, transient_pattern.matrices)
        ax_levels = np.hstack((pred_pattern.levels, transient_pattern.levels))

        if not isinstance(pred_pattern, EmptyPattern):
            depth, n_core = depth + 1, n_core + 1

            # Set core vertice's connections
            matrices = add_core_vertices(matrices, 1, n_core - 1)
            matrices['Cw'][0, -1], matrices['Cw'][1, -1] = 1, 1

            # Update output connection
            matrices['Ow'] = lil_matrix((n_core, pred_pattern.n_outputs))
            matrices['Ow'][-1, pred_pattern.output_id] = 1

            # Update levels
            ax_levels = np.hstack((ax_levels, np.array([2])))

        # Create partitions /!\ convention if transient always at index 0, pred at index 1 /!\
        partitions = [
            {'indices': [pred_pattern.size + j for j in range(transient_pattern.size)],
             "name": "transient", "depth": transient_pattern.depth},
            {'indices': range(pred_pattern.size), 'name': "pred", 'precision': pred_pattern.precision,
             "depth": pred_pattern.depth},
        ]

        # Add firing graph kwargs
        kwargs = {
            'partitions': partitions, 'precision': pred_pattern.precision, 'depth': depth, 'matrices': matrices,
            'ax_levels': ax_levels
        }

        return YalaDrainingPattern(
            pred_pattern.n_inputs, pred_pattern.n_outputs, pred_pattern.label_id, pred_pattern.output_id, **kwargs
        )

    @staticmethod
    def from_partition(partition, firing_graph, label_id=None, output_id=None, add_backward_firing=False):

        ax_levels = firing_graph.levels[partition['indices']]
        matrices = reduce_matrices(firing_graph.matrices, partition['indices'])
        label_id, output_id = partition.get('label_id', label_id), partition.get('output_id', output_id)

        # Add firing graph kwargs
        kwargs = {
            'partitions': partition['partitions'], 'precision': partition.get('precision', None),
            'depth': partition['depth'], 'matrices': matrices, 'ax_levels': ax_levels
        }

        if add_backward_firing:
            kwargs.update(
                {'backward_firing': reduce_backward_firing(firing_graph.backward_firing, partition['indices'])}
            )

        return YalaDrainingPattern(firing_graph.I.shape[0], firing_graph.O.shape[1], label_id, output_id, **kwargs)


class YalaDrainingPatterns(FiringGraph):

    def __init__(self, n_inputs, n_outputs, **kwargs):

        self.n_inputs, self.n_outputs = n_inputs, n_outputs
        kwargs.update({'project': 'YalaDrainingPatterns'})

        # Invoke parent constructor
        super(YalaDrainingPatterns, self).__init__(**kwargs)

    @staticmethod
    def check_patterns(l_patterns):
        assert all([isinstance(o, YalaDrainingPattern) for o in l_patterns]),\
            "Only YalaDrainingPattern can be used to build YalaDrainingPatters"

        assert len(set([o.depth for o in l_patterns])) == 1, \
            "Patterns of different depth inputed in YalaDrainingPatterns"

    @staticmethod
    def from_patterns(l_patterns):

        if len(l_patterns) == 0:
            return None

        YalaDrainingPatterns.check_patterns(l_patterns)

        l_partitions, l_levels, depth = [], [], l_patterns[0].depth
        n_core_current, n_in, n_out = 0, l_patterns[0].n_inputs, l_patterns[0].n_outputs
        d_matrices = create_empty_matrices(n_in, n_out, 0)

        for pattern in l_patterns:

            # Set partitions
            l_partitions.append({
                'indices': [n_core_current + i for i in range(pattern.Cw.shape[1])],
                'depth': pattern.depth,
                'label_id': pattern.label_id,
                'output_id': pattern.Ow.nonzero()[1][0]
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

        return YalaDrainingPatterns(n_in, n_out, **kwargs)


class YalaOutputSimplePattern(FiringGraph):
    """
    This class implement the main data structure used for fitting data. It is composed of weighted link in the form of
    scipy.sparse matrices and store complement information on vertices such as levels, mask for draining. It also keep
    track of the firing of vertices.

    """
    def __init__(self, n_inputs, n_outputs, **kwargs):

        self.n_inputs, self.n_outputs = n_inputs, n_outputs
        kwargs.update({'project': 'YalaOutputSimplePattern', 'depth': 2, 'ax_levels': np.ones(n_outputs)})

        # Invoke parent constructor
        super(YalaOutputSimplePattern, self).__init__(**kwargs)

    @staticmethod
    def from_mapping(d_mapping):

        n_inputs, n_outputs = len(d_mapping.keys()), sum([len(v) for v in d_mapping.values()])
        d_matrices, l_partitions = create_empty_matrices(n_inputs, n_outputs, n_outputs), []

        for k, v in d_mapping.items():
            d_matrices['Iw'][k, v], d_matrices['Ow'] = True, eye(n_outputs, format='csc')
            l_partitions.append({'indices': list(map(int, v)), 'index_input': int(k)})

        return YalaOutputSimplePattern(n_inputs, n_outputs, **{'partitions': l_partitions, 'matrices': d_matrices})

    def get_io_mapping(self):
        return {p['index_input']: p['indices'] for p in self.partitions}
