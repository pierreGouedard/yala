# Global imports
import numpy as np
from scipy.sparse import csc_matrix, diags, lil_matrix, hstack
from copy import deepcopy

# Local import
from src.model.patterns import YalaBasePatterns as ybp, YalaTopPattern as ytp
from src.model.data_models import BaseComponents, DrainerFeedbacks, DrainerParameters
from src.model.utils import set_feedbacks


class YalaPicker(object):
    """
    """
    def __init__(self, selector_type, mapping_feature_input, n_label, min_gain, min_precision, max_precision,
                 min_firing, margin):

        # type of picker
        self.selector_type = selector_type

        # characteristics of the picking problem
        self.map_fi, self.n_label = mapping_feature_input, n_label
        self.sum_fi = self.map_fi.sum(axis=0)

        # Keep track of completed patterns
        self.completes = None

        # Core picker parameter
        self.min_gain, self.min_precision, self.max_precision = min_gain, min_precision, max_precision
        self.min_firing, self.margin = min_firing, margin

    @staticmethod
    def concatenate_base_comp(l_comps):
        if all([c is None for c in l_comps]):
            return None

        # Get counts
        return BaseComponents(
            inputs=hstack([c.inputs for c in l_comps if c is not None]),
            levels=np.hstack([c.levels for c in l_comps if c is not None]),
            precisions=np.hstack([c.precisions for c in l_comps if c is not None]),
        )

    def set_drainer_params(self, partials):

        if partials is None:
            return None

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

            if components.transient_components is None:
                continue

            partials_, completes = self.pick_patterns(i, components.base_components, components.transient_components)

            # extend complete and partial patterns
            if partials_ is not None:
                partials = partials_.augment_from_pattern(partials, 'isolate')

            if completes is not None:
                self.completes = completes.augment_from_pattern(self.completes, 'same')

        # Create Top patterns to enable specific draining
        if partials is not None:
            server.pattern_backward = ytp.from_mapping(
                {k: [p['output_id'] for p in partials.partitions if p['label_id'] == k] for k in range(self.n_label)}
            )

        return partials, self.set_drainer_params(partials)

    def extract_completes_components(self, label_id):

        if self.completes is None:
            return None

        # Extract sub partitions
        l_partitions_sub = [p for p in self.completes.partitions if p['label_id'] == label_id]
        l_indices = [p['indices'][0] for p in l_partitions_sub]

        # Get components
        sax_inputs, ax_levels = self.completes.I[:, l_indices], self.completes.levels[l_indices]
        ax_precisions = np.array([p['precision'] for p in l_partitions_sub])

        # Remove extracted partitions from complete patterns
        self.completes = ybp.from_partitions(
            [p for p in self.completes.partitions if p['label_id'] != label_id], self.completes, 'label',
            n_label=self.n_label
        )

        return BaseComponents(inputs=sax_inputs, levels=ax_levels, precisions=ax_precisions)

    def get_sparse_feature_mask(self, sax_inputs):
        ax_feature_input = self.map_fi.astype(int).T.dot(sax_inputs).A
        ax_mask = ax_feature_input != self.sum_fi.T.A[:, [0] * ax_feature_input.shape[1]]
        ax_mask &= ax_feature_input > 0

        return ax_mask

    def merge_inputs(self, sax_left, sax_right, return_levels=False):
        # Get input mask left and right
        ax_left_mask, ax_right_mask = self.get_sparse_feature_mask(sax_left), self.get_sparse_feature_mask(sax_right)

        # Get base and trans dense inputs and compute
        sax_left_dense = sax_left + self.map_fi.dot(csc_matrix(~ax_left_mask * ax_right_mask))
        sax_right_dense = sax_right + self.map_fi.dot(csc_matrix(~ax_right_mask * ax_left_mask))

        # Compute levels feature based
        if return_levels:
            ax_feature_levels = ax_left_mask.sum(axis=0) + ax_right_mask.sum(axis=0)
            ax_feature_levels -= ax_left_mask.astype(int).transpose().dot(ax_right_mask).diagonal()

            return sax_left_dense.multiply(sax_right_dense), ax_feature_levels

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
        ax_trans_levels, ax_trans_precision,ax_trans_count = np.array([]), np.array([]), np.array([])
        l_inputs, n_base = [], 0
        for i in range(trans_comp.feature_precision.shape[1]):

            k = trans_comp.feature_precision[:, i].nnz
            if k == 0:
                n_base += 1
                continue

            # Extract best features and their input mask
            ax_features = trans_comp.feature_precision[:, i].A.ravel()
            ax_counts = trans_comp.feature_count[:, i].A.ravel()[ax_features > 0]
            sax_mask = self.map_fi.dot(diags(ax_features > 0, format='csc', dtype=bool)[:, ax_features > 0])

            # Extract inputs
            sax_inputs = sax_mask.multiply(trans_comp.inputs[:, [i] * k])

            if base_comp is not None:
                sax_inputs = self.merge_inputs(base_comp.inputs[:, [n_base] * k], sax_inputs)
                base_comp.reduce(np.setdiff1d(np.arange(base_comp.inputs.shape[1]), n_base))

            # Update levels
            ax_levels = (sax_inputs.T.dot(self.map_fi) > 0).sum(axis=1).A[:, 0]

            # Add candidates input and remove corresponding base pattern
            l_inputs.append(sax_inputs.copy())
            ax_trans_levels = np.hstack((ax_trans_levels, ax_levels))
            ax_trans_precision = np.hstack((ax_trans_precision, ax_features[ax_features > 0]))
            ax_trans_count = np.hstack((ax_trans_count, ax_counts))

        # Create component
        partial_comp, n_partial = None, 0
        if len(l_inputs):
            partial_comp = BaseComponents(
                inputs=hstack(l_inputs), levels=ax_trans_levels, precisions=ax_trans_precision,
                counts=ax_trans_count
            )
            n_partial = len(partial_comp)

        # Merge base component
        merged_comp = self.concatenate_base_comp([partial_comp, base_comp])
        if merged_comp is None:
            return None, None

        # Build candidate_patterns
        f_default = lambda x: dict(output_id=x, indices=[x], label_id=label_id, is_trans=x < n_partial)
        l_partitions = [dict(precision=p, count=c, **f_default(n)) for n, (p, c) in enumerate(merged_comp)]
        candidate_patterns = ybp.from_input_matrix(merged_comp.inputs, l_partitions, merged_comp.levels)

        return self.refine_picked_patterns(candidate_patterns)

    def refine_picked_patterns(self, cand_patterns):

        # Propagate activations
        sax_candidate = cand_patterns.propagate(self.sax_i)

        # Set variables for selection
        sax_selected = lil_matrix(sax_candidate.shape)
        ax_is_distinct = np.ones(sax_candidate.shape[1], dtype=bool)

        n, l_partials, l_completes = 0, [], []
        for d_score in sorted(cand_patterns.partitions, key=lambda x: x['precision'], reverse=True):
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
        partials_patterns = cand_patterns.from_partitions(l_partials, cand_patterns, 'isolate')
        complete_patterns = cand_patterns.from_partitions(l_completes, cand_patterns, 'label', n_label=self.n_label)

        return partials_patterns, complete_patterns


class YalaOrthogonalPicker(YalaPicker):

    """
    """
    def __init__(self, size, label_size, **kwargs):
        # Set specific attributes
        kwargs.update({'selector_type': 'orthogonal'})
        self.size, self.label_size = size, label_size

        # Invoke parent constructor
        super(YalaOrthogonalPicker, self).__init__(**kwargs)

    def filter_completes(self, complete_comp):
        if complete_comp is None:
            return None

        # Keep only complete components that reach the min_precision
        precision_filter = lambda x: x > self.min_precision
        idx_filtered = [i for i in range(len(complete_comp)) if precision_filter(complete_comp.precisions[i])]

        if len(idx_filtered):
            return BaseComponents(
                inputs=complete_comp.inputs[:, idx_filtered], levels=complete_comp.levels[idx_filtered],
                precisions=complete_comp.precisions[idx_filtered], counts=complete_comp.counts[idx_filtered]
            )

        else:
            return None

    def get_complement_attribute(self, base_comp, ind_comp, count, precision, label_id):

        if base_comp is not None:
            base_prec, base_count = base_comp.precisions[ind_comp], base_comp.counts[ind_comp]
        else:
            base_prec, base_count = self.label_size[label_id] / self.size, self.size

        c_count = count - base_count
        c_prec = (base_count * base_prec) - (count * precision) / c_count

        return c_prec, c_count

    def extract_comp(self, sax_inputs, ax_count, ax_precs, ax_levels):

        # Check for completion
        completion_check = lambda x: (ax_count[x] * ax_precs[x]) < self.min_firing

        # Get indices completes and partials
        idx_completes = [i for i in range(sax_inputs.shape[1]) if completion_check(i)]
        idx_partials = np.setdiff1d(np.arange(sax_inputs.shape[1]), idx_completes)

        # Build components of base patterns
        complete_comp, partial_comp = None, None
        if len(idx_completes):
            complete_comp = BaseComponents(
                inputs=sax_inputs[:, idx_completes], levels=ax_levels[idx_completes],
                precisions=ax_precs[idx_completes], counts=ax_count[idx_completes]
            )
        if len(idx_partials):
            partial_comp = BaseComponents(
                inputs=sax_inputs[:, idx_partials], levels=ax_levels[idx_partials], precisions=ax_precs[idx_partials],
                counts=ax_count[idx_partials]
            )

        return complete_comp, partial_comp

    def pick_patterns(self, label_id, base_comp, trans_comp):

        # Build input permutation to tackle ties
        ax_pertubations = np.random.randn(trans_comp.feature_precision.shape[0]) * 1e-6

        # build candidate input matrix
        ax_trans_levels, ax_trans_precs, ax_trans_count = np.array([]), np.array([]), np.array([])
        l_inputs, n_base = [], 0
        for i in range(len(trans_comp)):

            if trans_comp.feature_precision[:, i].nnz == 0:
                n_base += 1
                continue

            # Extract best cand precisions and their input mask
            ax_features = trans_comp.feature_precision[:, i].A.ravel() + ax_pertubations
            ax_count = trans_comp.feature_count[:, i].A.ravel()

            # Select best candidate
            max_idx =  ax_features.argmax()
            ax_precisions, ax_count = ax_features[[max_idx]], ax_count[[max_idx]]

            # Extract inputs
            sax_inputs = hstack((
                self.map_fi[:, max_idx].multiply(trans_comp.inputs[:, [i]]),
                self.map_fi[:, max_idx].astype(int) - trans_comp.inputs[:, [i]] > 0
            ))

            # Get levels and mask non valid input candidate
            ax_levels = (sax_inputs.sum(axis=0) > 0).A[0].astype(int)
            sax_inputs, ax_levels = sax_inputs[:, ax_levels > 0], ax_levels[ax_levels > 0]

            # Merge base and trans inputs, update orthogonal complement precision
            if base_comp is not None:
                # Merge candidate input with base comp inputs and infer complement precision
                sax_inputs, ax_f_levels = self.merge_inputs(
                    base_comp.inputs[:, [n_base] * len(ax_levels)], sax_inputs, True
                )

                # Recompute levels and mask of non valid inputs
                ax_levels = (sax_inputs.T.dot(self.map_fi) > 0).sum(axis=1).A[:, 0]
                sax_inputs, ax_levels = sax_inputs[:, (ax_levels == ax_f_levels)], ax_levels[(ax_levels == ax_f_levels)]

                # Remove base component
                base_comp.reduce(np.setdiff1d(np.arange(base_comp.inputs.shape[1]), n_base))

            # Compute complement count and precision
            if len(ax_levels) > 1:
                precision, count = self.get_complement_attribute(
                    base_comp, n_base, ax_count[0], ax_precisions[0], label_id
                )
                ax_precisions, ax_count = np.hstack([ax_precisions, [precision]]), np.hstack([ax_count, [count]])

            # Update trans components
            ax_trans_levels = np.hstack([ax_trans_levels, ax_levels])
            ax_trans_precs = np.hstack([ax_trans_precs, ax_precisions])
            ax_trans_count = np.hstack([ax_trans_count, ax_count])
            l_inputs.append(sax_inputs.copy())

        # Extract components
        complete_comp, partial_comp = None, None
        if l_inputs:
            complete_comp, partial_comp = self.extract_comp(
                hstack(l_inputs), ax_trans_count, ax_trans_precs, ax_trans_levels
            )

        # filter base_comp on there precision.
        complete_comp = self.filter_completes(self.concatenate_base_comp([complete_comp, base_comp]))

        # Build complete patterns
        completes = None
        if complete_comp:
            f_default = lambda x: dict(indices=[x], label_id=label_id, output_id=label_id)
            l_parts = [dict(precision=p, count=c, **f_default(n)) for n, (p, c) in enumerate(complete_comp)]
            completes = ybp.from_input_matrix(complete_comp.inputs, l_parts, complete_comp.levels, self.n_label)

        # Build partials patterns
        partials = None
        if partial_comp:
            f_default = lambda x: dict(indices=[x], label_id=label_id, output_id=x)
            l_partitions = [dict(precision=p, count=c, **f_default(n)) for n, (p, c) in enumerate(partial_comp)]
            partials = ybp.from_input_matrix(partial_comp.inputs, l_partitions, partial_comp.levels)

        return partials, completes
