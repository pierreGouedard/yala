# Global imports
import numpy as np
from scipy.sparse import lil_matrix, hstack, eye

# Local import
from firing_graph.data_structure.graph import FiringGraph
from firing_graph.data_structure.utils import create_empty_matrices, add_core_vertices, reduce_matrices, \
    reduce_backward_firing, augment_matrices, set_matrices_spec


class EmptyPattern(FiringGraph):

    def __init__(self, n_inputs, n_outputs, label_id, output_id, **kwargs):

        self.n_inputs, self.n_outputs, self.size = n_inputs, n_outputs, 0
        self.label_id, self.output_id = label_id, output_id
        kwargs.update({
            'project': 'EmptyPattern', 'ax_levels': np.array([]), 'depth': 0, 'precision': 0,
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
        self.label_id, self.output_id = label_id, output_id

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
        label_id = label_id if label_id is not None else partition['label_id']
        output_id = output_id if output_id is not None else partition['output_id']
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
        sax_O = lil_matrix((1, self.n_outputs))
        sax_O[0, output_id] = 1
        self.matrices['Ow'], self.matrices['Om'] = sax_O.copy(), sax_O.copy()

        # Set matrices format and type
        set_matrices_spec(self.matrices, write_mode=False)

        # Change index output
        self.output_id = output_id

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
    def from_input_matrix(sax_I, l_partitions, ax_levels, n_outputs=None):

        assert sax_I.shape[1] == len(l_partitions), "The # core vertices is not equal to the length of partitions"

        if len(l_partitions) == 0:
            return None

        # Get output indices and intialize matrices
        t_outs = tuple(d_part['output_id'] for d_part in l_partitions)
        n_outputs = max(t_outs) + 1 if n_outputs is None else n_outputs
        d_matrices = create_empty_matrices(
            n_inputs=sax_I.shape[0], n_outputs=n_outputs, n_core=sax_I.shape[1]
        )

        # Set matrices
        d_matrices['Iw'] = sax_I
        d_matrices['Ow'][range(sax_I.shape[1]), t_outs] = 1

        # Add firing graph kwargs
        kwargs = {'partitions': l_partitions, 'matrices': d_matrices, 'ax_levels': ax_levels}

        return YalaBasePatterns(d_matrices['Iw'].shape[0], d_matrices['Ow'].shape[1], **kwargs)

    @staticmethod
    def from_patterns(l_base_patterns, output_method, **kwargs):
        assert output_method in ["isolate", "label", "same"], f"Unknown output_method {output_method}"

        if len(l_base_patterns) == 0:
            return None

        # Set Input matrix and levels from patterns
        sax_I = hstack([p.I for p in l_base_patterns])
        ax_levels = np.array([p.levels[0] for p in l_base_patterns])

        # Set partitions
        l_partitions = [dict(
            indices=[i], depth=2, output_id={'isolate': i, 'label': p.label_id}.get(output_method, p.output_id),
            precision=p.precision, score=p.score, label_id=p.label_id, **kwargs
        ) for i, p in enumerate(l_base_patterns)]

        # Set number of output
        n_outputs = None
        if output_method in ['same', 'label']:
            n_outputs = l_base_patterns[0].O.shape[1]

        return YalaBasePatterns.from_input_matrix(sax_I, l_partitions, ax_levels, n_outputs=n_outputs)

    @staticmethod
    def from_partitions(l_partitions, firing_graph, output_method, **kwargs):
        assert output_method in ["isolate", "label", "same"], f"Unknown output_method {output_method}"

        if len(l_partitions) == 0:
            return None

        # Extract core vertices input and level of interest
        sax_I = firing_graph.I[:, sum([p['indices'] for p in l_partitions], [])]
        ax_levels = firing_graph.levels[sum([p['indices'] for p in l_partitions], [])]

        # Update partitions
        l_partitions = [dict(
            indices=[i], depth=2, output_id={'isolate': i, 'label': p['label_id']}.get(output_method, p['output_id']),
            precision=p['precision'], score=p['score'], label_id=p['label_id'], **kwargs
        ) for i, p in enumerate(l_partitions)]

        # Set number of output
        n_outputs = None
        if output_method in ['same', 'label']:
            n_outputs = firing_graph.O.shape[1]

        return YalaBasePatterns.from_input_matrix(sax_I, l_partitions, ax_levels, n_outputs=n_outputs)

    def remove(self, index,  output_method='isolate', **kwargs):
        l_partitions = [p for p in self.partitions if p['indices'][0] != index]
        return self.from_partitions(l_partitions, self, output_method, **kwargs)

    def augment_from_patterns(self, l_base_patterns, output_method, **kwargs):

        if len(l_base_patterns) == 0:
            return self

        # Create necessary variables
        sax_I = hstack([pattern.I for pattern in l_base_patterns])
        l_partitions = [
            {'precision': p.precision, 'score': p.score, 'label_id': p.label_id, 'output_id': p.output_id}
            for p in l_base_patterns
        ]
        ax_levels = np.array([p.levels[0] for p in l_base_patterns])

        return self.augment_from_inputs(sax_I, l_partitions, ax_levels, output_method, **kwargs)

    def augment_from_inputs(self, sax_I, l_partitions, ax_levels, output_method, **kwargs):
        assert output_method in ["isolate", "label", "same"], f"Unknown output_method {output_method}"

        # Compute new input matrix and partitions
        sax_I = hstack([self.I, sax_I])
        l_partitions = self.partitions + [dict(
            indices=[self.C.shape[0] + i], depth=2, precision=p['precision'], score=p['score'], label_id=p['label_id'],
            output_id={'isolate': i, 'label': p['label_id']}.get(output_method, p['output_id']), **kwargs
        ) for i, p in enumerate(l_partitions)]

        # Set number of output
        n_outputs = None
        if output_method in ['same', 'label']:
            n_outputs = self.O.shape[1]

        return YalaBasePatterns.from_input_matrix(sax_I, l_partitions, np.hstack((self.levels, ax_levels)), n_outputs)

    def reset_output(self, key='label_id', l_outputs=None):
        if l_outputs is None:
            l_outputs = [p[key] for p in self.partitions]

        # Init output matrix
        self.matrices['Ow'] = lil_matrix((self.Ow.shape[0], max(l_outputs) + 1))
        self.matrices['Om'] = lil_matrix((self.Om.shape[0], max(l_outputs) + 1))

        for i, p in enumerate(self.partitions):
            p['output_id'] = l_outputs[i]
            self.matrices['Ow'][p['indices'], p['output_id']] = True
            self.matrices['Om'][p['indices'], p['output_id']] = True

        # Set matrices format and type
        set_matrices_spec(self.matrices, write_mode=False)

        return self

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
    def from_patterns(base, transient):

        depth, n_core = 2, 2
        matrices = augment_matrices(base.matrices, transient.matrices)
        ax_levels = transient.levels

        if not isinstance(base, EmptyPattern):
            depth, n_core = depth + 1, n_core + 1

            # Set core vertice's connections
            matrices = add_core_vertices(matrices, 1, n_core - 1)
            matrices['Cw'][0, -1], matrices['Cw'][1, -1] = 1, 1

            # Update output connection
            matrices['Ow'] = lil_matrix((n_core, base.n_outputs))
            matrices['Ow'][-1, base.output_id] = 1

            # Update levels
            ax_levels = np.hstack((base.levels, ax_levels, np.array([2])))

        # Add firing graph kwargs
        kwargs = {
            'precision': base.precision, 'depth': depth, 'matrices': matrices, 'ax_levels': ax_levels,
            'score': base.score
        }

        return YalaDrainingPattern(base.n_inputs, base.n_outputs, base.label_id, base.output_id, **kwargs)


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

            # Uodate partitions
            l_partitions.append({
                'indices': [n_core_current + i for i in range(pattern.Cw.shape[1])],
                'depth': pattern.depth,
                'label_id': pattern.label_id,
                'output_id': pattern.Ow.nonzero()[1][0],
                'precision': pattern.precision,
                'score': pattern.score
            })

            n_core_current += pattern.Cw.shape[1]
            d_matrices = augment_matrices(d_matrices, pattern.matrices, write_mode=None)

            # Merge levels
            l_levels.extend(list(pattern.levels))

        # Add firing graph kwargs
        kwargs = {'partitions': l_partitions, 'ax_levels': np.array(l_levels), 'matrices': d_matrices, 'depth': depth}

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
