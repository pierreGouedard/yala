# Global imports
import numpy as np
from scipy.sparse import csc_matrix, diags, lil_matrix, hstack
from copy import deepcopy

# Local import
from .patterns import YalaBasePatterns, YalaTopPattern
from .data_models import BaseComponents, DrainerFeedbacks, DrainerParameters
from .utils import set_feedbacks


class YalaPicker(object):
    """
    """
    def __init__(self, selector_type, mapping_feature_input, n_label, min_gain, min_precision, max_precision,
                 min_firing, margin):

        # type of picker
        self.selector_type = selector_type

        # characteristics of the picking problem
        self.map_fi, self.n_label = mapping_feature_input, n_label

        # Keep track of completed patterns
        self.completes = None

        # Core picker parameter
        self.min_gain, self.min_precision, self.max_precision = min_gain, min_precision, max_precision
        self.min_firing, self.margin = min_firing, margin

    @staticmethod
    def concatenate_base_comp(l_comps):
        if all([c is None for c in l_comps]):
            return None

        return BaseComponents(
            inputs=hstack([c.inputs for c in l_comps if c is not None]),
            levels=np.hstack((c.levels for c in l_comps if c is not None)),
            precisions=np.hstack((c.precisions for c in l_comps if c is not None))
        )

    def set_drainer_params(self, partials):

        # Get targeted precision for each output
        ax_precision = np.array([p['precision'] for p in sorted(partials.partitions, key=lambda x: x['output_id'])])
        ax_lower_precision = (ax_precision - (2 * self.margin)).clip(min=self.min_gain)
        ax_upper_precision = (ax_precision - self.margin).clip(min=2 * self.min_gain)

        # Compute drainer params: penalties, rewards and initial weights
        ax_p, ax_r = set_feedbacks(ax_lower_precision, ax_upper_precision)
        drainer_params = DrainerParameters(
            feedbacks=DrainerFeedbacks(penalties=ax_p, rewards=ax_r),
            weights=((ax_p - (ax_lower_precision * (ax_p + ax_r))) * self.min_firing).astype(int) + 1
        )

        return drainer_params

    def pick_patterns_multi_label(self, server, firing_graph):
        partials, n = None, 0
        for i in range(self.n_label):
            components = firing_graph.extract_drained_components(i, self.min_firing, self.min_gain, self.map_fi)

            if components.trans_components is None:
                continue

            partials_, completes = self.pick_patterns(i, components.base_components, components.trans_components)

            # extend complete and partial patterns
            if partials_ is not None:
                partials = partials_.augment_from_pattern(partials, 'isolate')

            if completes is not None:
                self.completes = completes.augment_from_pattern(self.completes, 'same')

        # Create Top patterns to enable specific draining
        server.pattern_backward = YalaTopPattern.from_mapping(
            {k: [p.output_id for p in partials.partitions if p.label_id == k] for k in range(self.n_label)}
        )

        return partials, self.set_drainer_params(partials)

    def extract_completes_components(self, label_id):

        if self.completes is None:
            return None

        # Extract sub partitions
        l_partitions_sub = [p for p in self.completes.partitions if p['label_id'] == label_id]
        l_indices = [p['indices'] for p in l_partitions_sub]

        # Get components
        sax_inputs, ax_levels = self.completes.I[:, l_indices], self.completes.levels[l_indices]
        ax_precisions = np.array([p['precision'] for p in l_partitions_sub])

        # Remove extracted partitions from complete patterns
        self.completes = YalaBasePatterns.from_partitions(
            [p for p in self.completes if p.label_id != label_id], self.completes, 'label'
        )

        return BaseComponents(inputs=sax_inputs, levels=ax_levels, precisions=ax_precisions)

    def merge_inputs(self, sax_left, sax_right):
        # Get input mask left and right
        ax_left_mask, ax_right_mask = self.map_fi.T.dot(sax_left).A > 0, self.map_fi.T.dot(sax_right).A > 0

        # Get base and trans dense inputs and compute
        sax_left_dense = sax_left + self.map_fi.dot(csc_matrix(~ax_left_mask * ax_right_mask))
        sax_right_dense = sax_right + self.map_fi.dot(csc_matrix(~ax_right_mask * ax_left_mask))

        return sax_left_dense.multiply(sax_right_dense)

    def pick_patterns(self, label_id, base_comp, trans_comp):
        raise NotImplementedError


class YalaGreedyPicker(YalaPicker):
    """
    """
    def __init__(self, sax_i, n_overlap, **kwargs):

        # Set specific attributes
        self.sax_i, self.n_overlap = sax_i, n_overlap

        # Invoke parent constructor
        kwargs.update({'selector_type': 'greedy'})
        super(YalaGreedyPicker, self).__init__(**kwargs)

    @staticmethod
    def update_overlap_mask(sax_selected, sax_candidates, n_overlap):
        ax_candidates_norm = sax_candidates.astype(int).sum(axis=0).A[0]
        ax_inner_product = csc_matrix(sax_selected.sum(axis=1)).transpose().astype(int).dot(sax_candidates).A[0]
        return (ax_candidates_norm - ax_inner_product) > n_overlap

    def pick_patterns(self, label_id, base_comp, trans_comp):

        # Extract previously completes patterns
        if base_comp is not None:
            base_comp = self.concatenate_base_comp([base_comp, self.extract_completes_components(label_id)])

        # Extract partials components
        ax_trans_levels, ax_trans_precision, l_inputs, n_base = np.array([]), np.array([]), [], 0
        for i in range(trans_comp.feature_precision.shape[1]):

            k = trans_comp.feature_precision[:, i].nnz
            if k == 0:
                n_base += 1
                continue

            # Extract best features and their input mask
            ax_features = trans_comp.feature_precision[:, i].A.ravel()
            sax_mask = self.map_fi.dot(diags(ax_features > 0, format='csc', dtype=bool)[:, ax_features > 0])

            # Extract inputs
            sax_inputs = sax_mask.multiply(trans_comp.inputs[:, [i] * k])

            if base_comp is not None:
                sax_inputs = self.merge_inputs(base_comp.inputs.I[:, [n_base] * k], sax_inputs)
                ax_indices = np.setdiff1d(np.arange(base_comp.inputs.shape[1]), n_base)
                base_comp.reduce(ax_indices)

            # Update levels
            ax_levels = (sax_inputs.T.dot(self.map_fi) > 0).sum(axis=1).A[:, 0]

            # Add candidates input and remove corresponding base pattern
            l_inputs.append(sax_inputs.copy())
            ax_trans_levels = np.hstack((ax_trans_levels, ax_levels))
            ax_trans_precision = np.hstack((ax_trans_precision, ax_features[ax_features > 0]))

        # Create component
        partial_comp = BaseComponents(inputs=hstack(l_inputs), levels=ax_trans_levels, precisions=ax_trans_precision)

        # Merge base component
        merged_comp = self.concatenate_base_comp([partial_comp, base_comp])

        # Build candidate_patterns
        f_default = lambda x: dict(output_id=x, indices=[x], label_id=label_id, is_trans=x < len(partial_comp))
        l_partitions = [dict(precision=p, **f_default(n)) for n, p in enumerate(merged_comp.precisions)]
        candidate_patterns = YalaBasePatterns.from_input_matrix(merged_comp.inputs, l_partitions, merged_comp.levels)

        return self.refine_picked_patterns(candidate_patterns)

    def refine_picked_patterns(self, candidate_patterns):

        # Propagate activations
        sax_candidate = candidate_patterns.propagate(self.sax_i)

        # Set variables for selection
        sax_selected = lil_matrix(sax_candidate.shape)
        ax_is_distinct = np.ones(sax_candidate.shape[1], dtype=bool)

        n, l_partials, l_completes = 0, [], []
        for d_score in sorted(candidate_patterns.partitions, key=lambda x: x['precision'], reverse=True):
            if not ax_is_distinct[d_score['output_id']]:
                continue

            # if target precision of a base pattern is not reached, drop the pattern
            if not d_score['is_trans']:
                if d_score['precision'] < self.min_precision:
                    continue

            # Update variables
            sax_selected[:, n] = sax_candidate[:, d_score['output_id']] > 0
            ax_is_distinct = self.update_overlap_mask(sax_selected, sax_candidate, self.n_overlap)

            # update partition lists
            if d_score['is_trans']:
                l_partials.append(d_score)
            else:
                l_completes.append(d_score)

            n += 1

        # Gte partial and complete patterns
        partials_patterns = candidate_patterns.from_partitions(l_partials, candidate_patterns, 'isolate')
        complete_patterns = candidate_patterns.from_partitions(l_completes, candidate_patterns, 'label')

        return partials_patterns, complete_patterns


class YalaOrthogonalPicker(YalaPicker):

    """
    """
    def __init__(self, k, **kwargs):
        # Set specific attributes
        self.k = k
        kwargs.update({'selector_type': 'orthogonal'})

        # Invoke parent constructor
        super(YalaOrthogonalPicker, self).__init__(**kwargs)

    @staticmethod
    def generate_coefficients(k):
        ax_upper = np.hstack((np.eye(k, dtype=bool), np.zeros((k, 1), dtype=bool)))
        ax_lower = np.hstack((np.zeros((k, 1), dtype=bool), np.triu(np.ones((k, k), dtype=bool), 0)))
        return csc_matrix(np.vstack((ax_upper, ax_lower)))

    def pick_patterns(self, label_id, base_comp, trans_comp):

        # Build input permutation to tackle ties
        ax_pertubations = np.random.randn(trans_comp.feature_precision.shape[0]) * 1e-6

        # build candidate input matrix
        ax_trans_levels, ax_trans_precision, ax_trans_count = np.array([]), np.array([]), np.array([])
        l_inputs, n_base = [], 0
        for i in range(trans_comp.feature_precision.shape[1]):

            k = min(self.k, trans_comp.feature_precision[:, i].nnz)
            if k == 0:
                n_base += 1
                continue

            # Extract best features and their input mask
            ax_count = trans_comp.feature_count[:, i].A.ravel()
            ax_features = trans_comp.feature_precision[:, i].A.ravel() + ax_pertubations
            ax_features = ax_features * (ax_features >= np.sort(ax_features)[-k])
            sax_mask = self.map_fi.dot(
                diags(ax_features > 0, format='csc', dtype=bool)[:, ax_features.argsort()[:-k - 1:-1]]
            )

            # Extract inputs
            sax_inputs = hstack((
                sax_mask.multiply(trans_comp.inputs[:, [i] * k]),
                sax_mask.astype(int) - trans_comp.inputs[:, [i] * k] > 0
            )).dot(self.generate_coefficients(k))

            # Merge base and trans inputs and remove base pattern if necessary
            if base_comp is not None:
                sax_inputs = self.merge_inputs(base_comp.inputs[:, [n_base] * (k + 1)], sax_inputs)
                ax_indices = np.setdiff1d(np.arange(base_comp.inputs.shape[1]), n_base)
                base_comp.reduce(ax_indices)

            # Compute levels and precision
            ax_levels = (sax_inputs.T.dot(self.map_fi) > 0).sum(axis=1).A[:, 0]
            ax_precisions = np.hstack((ax_features[ax_features.argsort()[:-k - 1:-1]], [1 - ax_features.max()]))
            ax_count = np.hstack((ax_count[ax_features.argsort()[:-k - 1:-1]], [np.inf]))

            # Update trans components
            ax_trans_levels = np.hstack([ax_trans_levels, ax_levels])
            ax_trans_precision = np.hstack([ax_trans_precision, ax_precisions])
            ax_trans_count = np.hstack([ax_trans_count, ax_count])
            l_inputs.append(sax_inputs.copy())

        # Build complete patterns
        f_default = lambda x: dict(indices=[x], label_id=label_id, output_id=label_id)
        l_partitions = [dict(precision=p, **f_default(n)) for n, p in enumerate(base_comp.precisions)]
        completes = YalaBasePatterns.from_input_matrix(base_comp.inputs, l_partitions, base_comp.levels)

        # Build and refine partials patterns
        # TODO: remove partitions for which (ax_trans_count * prec / min_prec) < min_firing and add them to completes
        partials = YalaBasePatterns.from_input_matrix(hstack(l_inputs), l_partitions, ax_trans_levels)

        return partials, completes
