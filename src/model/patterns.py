# Global imports
import numpy as np
from scipy.sparse import lil_matrix, hstack, vstack, eye, diags, csc_matrix
from dataclasses import dataclass

# Local import
from firing_graph.data_structure.graph import FiringGraph
from firing_graph.data_structure.utils import create_empty_matrices, set_matrices_spec
from .data_models import TransientComponents, BaseComponents, ExtractedDrainedComponents


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

        # Get output indices and initialize matrices
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
    def from_partitions(l_partitions, firing_graph, output_method, n_label=None, **kwargs):
        assert output_method in ["isolate", "label", "same"], f"Unknown output_method {output_method}"

        if len(l_partitions) == 0:
            return None

        # Extract core vertices input and level of interest
        sax_I = firing_graph.I[:, sum([p['indices'] for p in l_partitions], [])]
        ax_levels = firing_graph.levels[sum([p['indices'] for p in l_partitions], [])]

        # Update partitions
        l_partitions = [dict(
            indices=[i], depth=2, output_id={'isolate': i, 'label': p['label_id']}.get(output_method, p['output_id']),
            precision=p['precision'], count=p['count'], label_id=p['label_id'], **kwargs
        ) for i, p in enumerate(l_partitions)]

        # Set number of output
        n_outputs = None
        if output_method == 'same':
            n_outputs = firing_graph.O.shape[1]

        elif output_method == 'label':
            n_outputs = n_label

        return YalaBasePatterns.from_input_matrix(sax_I, l_partitions, ax_levels, n_outputs)

    def augment_from_pattern(self, pattern, output_method, **kwargs):
        if pattern is None:
            return self

        return self.augment_from_inputs(pattern.I, pattern.partitions, pattern.levels, output_method, **kwargs)

    def augment_from_inputs(self, sax_I, l_partitions, ax_levels, output_method, n_label=None, **kwargs):
        assert output_method in ["isolate", "label", "same"], f"Unknown output_method {output_method}"

        # Compute new input matrix and partitions
        sax_I = hstack([self.I, sax_I])
        l_partitions = self.partitions + [dict(
            indices=[self.C.shape[0] + i], depth=2, precision=p['precision'], count=p['count'], label_id=p['label_id'],
            output_id={'isolate': self.C.shape[0] + i, 'label': p['label_id']}.get(output_method, p['output_id']),
            **kwargs
        ) for i, p in enumerate(l_partitions)]

        # Set number of output
        n_outputs = self.n_outputs
        if output_method == 'isolate':
            n_outputs = None

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


class YalaDrainingPatterns(FiringGraph):

    def __init__(self, n_inputs, n_outputs, drainer_params, **kwargs):

        self.n_inputs, self.n_outputs = n_inputs, n_outputs
        self.drainer_params = drainer_params
        kwargs.update({'project': 'YalaDrainingPatterns'})

        # Invoke parent constructor
        super(YalaDrainingPatterns, self).__init__(**kwargs)

    @staticmethod
    def get_precision(sax_weight, sax_count, ax_p, ax_r, ax_w, n0, ax_prec, map_fi):

        # Get input weights and count
        sax_i_mask = (sax_weight > 0).multiply(sax_count > 0)
        sax_i_count = map_fi.transpose().astype(int).dot(sax_i_mask)

        # Derive feature weight, count and mask
        sax_f_weight = map_fi.transpose().dot(sax_weight.multiply(sax_i_mask))
        sax_f_count = map_fi.transpose().dot(sax_count.multiply(sax_i_mask))
        sax_f_mask = (sax_f_weight > 0).multiply(sax_f_count >= n0)

        # TODO: test: computation by input (not by features)
        sax_nom = sax_weight.multiply(sax_i_mask) - sax_i_mask.dot(diags(ax_w, format='csc'))
        sax_denom = sax_i_mask.multiply(sax_count.dot(diags(ax_p + ax_r, format='csc')))
        sax_precision_i = sax_nom.multiply(sax_denom.astype(float).power(-1))
        sax_precision_i += (sax_precision_i != 0).dot(diags(ax_p / (ax_p + ax_r), format='csc'))

        # compute precision: (original formula: float(score - weight) / (t * (p + r)) + float(p) / (p + r))
        sax_nom = sax_f_weight.multiply(sax_f_mask) - sax_f_mask.multiply(sax_i_count).dot(diags(ax_w, format='csc'))
        sax_denom = sax_f_mask.multiply(sax_f_count.dot(diags(ax_p + ax_r, format='csc')))
        sax_precision = sax_nom.multiply(sax_denom.astype(float).power(-1))
        sax_precision += (sax_precision != 0).dot(diags(ax_p / (ax_p + ax_r), format='csc'))

        # print('prec')
        # import IPython
        # IPython.embed()

        # Get only precision mask that are larger than target precision (ax_prec)
        precision_mask = sax_precision > (sax_precision > 0).dot(diags(ax_prec, format='csc'))
        sax_precision, sax_count = sax_precision.multiply(precision_mask), sax_f_count.multiply(precision_mask)

        return sax_precision, sax_count, sax_i_mask.multiply(map_fi.dot(precision_mask))

    @staticmethod
    def from_input_matrix(sax_inputs, drainer_params, n_label, base_patterns=None):

        n_inputs, n_trans = sax_inputs.shape[0], sax_inputs.shape[1]
        if base_patterns is None:
            # Set matrices
            d_matrices = create_empty_matrices(n_inputs, n_trans, n_trans)
            d_matrices['Iw'], d_matrices['Im'] = sax_inputs, sax_inputs > 0
            d_matrices['Ow'] = diags(np.ones(n_trans, dtype=bool), dtype=bool, format='csc')

            # Set partitions and others
            l_partitions = [
                {'indices': [i], 'label_id': i, 'output_id': i, 'precision': 0, 'count': 0} for i in range(n_trans)
            ]
            ax_levels, depth = np.ones(n_trans), 2

            # Set forward mask relation
            sax_I_mask = csc_matrix(np.eye(n_trans))

        else:
            # set inputs
            d_matrices = create_empty_matrices(n_inputs, n_trans, n_trans * 3)
            d_matrices['Iw'] = hstack([base_patterns.I, sax_inputs, csc_matrix(sax_inputs.shape)])
            d_matrices['Im'] = hstack([
                csc_matrix(sax_inputs.shape, dtype=bool), sax_inputs > 0, csc_matrix(sax_inputs.shape, dtype=bool)
            ])

            # Set layer 1 -> layer 2 links
            sax_c_layer_1 = hstack([csc_matrix((n_trans, 2 * n_trans)), csc_matrix(np.eye(n_trans))])
            d_matrices['Cw'] = vstack([sax_c_layer_1, sax_c_layer_1, csc_matrix((n_trans, 3 * n_trans))])

            # Set output links
            l_idx_out = [p['output_id'] for p in sorted(base_patterns.partitions, key=lambda x: x['indices'][0])]
            d_matrices['Ow'] = vstack([csc_matrix((2 * n_trans, n_trans)), csc_matrix(np.eye(n_trans))[l_idx_out, :]])

            # Set additional
            l_partitions = [
                dict(indices=[p['indices'][0], p['indices'][0] + n_trans, p.pop('indices')[0] + (2 * n_trans)], **p)
                for p in base_patterns.partitions
            ]
            ax_levels, depth = np.hstack([base_patterns.levels, np.ones(n_trans), np.ones(n_trans) * 2]), 3

            # Set forward mask relation
            sax_I_mask = lil_matrix((n_label, d_matrices['Cw'].shape[1]), dtype=bool)
            for p in l_partitions:
                sax_I_mask[p['label_id'], p['indices'][:-1]] = True
            sax_I_mask = sax_I_mask

        # Add firing graph kwargs
        kwargs = {
            'depth': depth, 'matrices': d_matrices, 'ax_levels': ax_levels, 'score': 0, 'partitions': l_partitions,
            "I_mask": sax_I_mask
        }

        return YalaDrainingPatterns(n_inputs, n_trans, drainer_params, **kwargs)

    def extract_drainer_params(self, idx):
        ax_p, ax_r = self.drainer_params.feedbacks.penalties[idx], self.drainer_params.feedbacks.rewards[idx]
        ax_weights = self.drainer_params.weights[idx]

        return ax_p, ax_r, ax_weights

    def extract_drained_components(self, label_id, min_firing, min_gain, map_fi):

        # Get sub partitions
        l_parts = [p for p in self.partitions if p['label_id'] == label_id]

        if not len(l_parts):
            return ExtractedDrainedComponents(base_components=None, transient_components=None)

        index_trans = 1 if len(l_parts[0]['indices']) == 3 else 0

        # Get precision and drainer params
        ax_precisions, ax_counts = np.array([p['precision'] for p in l_parts]), np.array([p['count'] for p in l_parts])
        ax_p, ax_r, ax_weights = self.extract_drainer_params([p['output_id'] for p in l_parts])

        # Extract TransComponents
        sax_weights = self.Iw[:, [p['indices'][index_trans] for p in l_parts]].astype(float)
        sax_counts = self.backward_firing['i'][:, [p['indices'][index_trans] for p in l_parts]].astype(float)
        sax_features_prec, sax_features_count, sax_trans_input = self.get_precision(
            sax_weights, sax_counts, ax_p, ax_r, ax_weights, min_firing, ax_precisions + min_gain, map_fi
        )
        trans_components = TransientComponents(
            feature_precision=sax_features_prec, feature_count=sax_features_count, inputs=sax_trans_input
        )

        # Extract BaseComponents
        if index_trans == 1:
            base_components = BaseComponents(
                inputs=self.I[:, [p['indices'][0] for p in l_parts]],
                levels=self.levels[[p['indices'][0] for p in l_parts]], precisions=ax_precisions, counts=ax_counts
            )

        else:
            base_components = None

        return ExtractedDrainedComponents(base_components=base_components, transient_components=trans_components)


class YalaTopPattern(FiringGraph):
    """
    This class implement the main data structure used for fitting data. It is composed of weighted link in the form of
    scipy.sparse matrices and store complement information on vertices such as levels, mask for draining. It also keep
    track of the firing of vertices.

    """
    def __init__(self, n_inputs, n_outputs, **kwargs):

        self.n_inputs, self.n_outputs = n_inputs, n_outputs
        kwargs.update({'project': 'YalaTopPattern', 'depth': 2, 'ax_levels': np.ones(n_outputs)})

        # Invoke parent constructor
        super(YalaTopPattern, self).__init__(**kwargs)

    @staticmethod
    def from_mapping(d_mapping):

        n_inputs, n_outputs = len(d_mapping.keys()), sum([len(v) for v in d_mapping.values()])
        d_matrices, l_partitions = create_empty_matrices(n_inputs, n_outputs, n_outputs), []

        for k, v in d_mapping.items():
            d_matrices['Iw'][k, v], d_matrices['Ow'] = True, eye(n_outputs, format='csc')
            l_partitions.append({'indices': list(map(int, v)), 'index_input': int(k)})

        return YalaTopPattern(n_inputs, n_outputs, **{'partitions': l_partitions, 'matrices': d_matrices})

    def get_io_mapping(self):
        return {p['index_input']: p['indices'] for p in self.partitions}
