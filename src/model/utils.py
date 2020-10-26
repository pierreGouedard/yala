# Global imports
import numpy as np
from scipy.sparse import csc_matrix, diags, lil_matrix, hstack

# Local import
from .patterns import EmptyPattern, YalaBasePattern, YalaDrainingPattern, \
    YalaDrainingPatterns, YalaBasePatterns, YalaOutputSimplePattern


def build_firing_graph(sampler, ax_weights, l_transients=None, n_inputs=None):
    """

    :param sampler:
    :param weights:
    :return:
    """

    if sampler.samples is None:
        raise ValueError(
            "Before Building firing graph, one need to sample input bits using generative or discriminative "
            "sampling"
        )

    l_patterns = []
    if l_transients is None:
        n_outputs = len(ax_weights)
        for i in range(n_outputs):
            # Add Empty base and sampled intersection into a Yala structure
            l_patterns.append(YalaDrainingPattern.from_patterns(
                EmptyPattern(n_inputs, n_outputs, i, i),
                YalaBasePattern.from_input_indices(n_inputs, n_outputs, i, i, sampler.samples[i], 1, ax_weights[i])
            ))

    else:
        for i, pattern in enumerate(l_transients):
            # Add Empty base and sampled intersection into a Yala structure
            l_patterns.append(YalaDrainingPattern.from_patterns(
                pattern,
                YalaBasePattern.from_input_indices(
                    pattern.n_inputs, pattern.n_outputs, pattern.label_id, pattern.output_id, sampler.samples[i], 1,
                    ax_weights[pattern.output_id]
                )
            ))

    # Replace by YalaDrainingPatterns.from_patterns
    firing_graph = YalaDrainingPatterns.from_patterns(l_patterns)

    return firing_graph


def get_normalized_precision(sax_activations, ax_precision, ax_new_mask):

    # If no candidate for norm return empty list
    if not ax_new_mask.any():
        return []

    # If only one candidate for norm return un-changed precision
    if sax_activations.shape[-1] == 1:
        return ax_precision

    # Build diff operator
    n = sax_activations.shape[-1] - 1
    sax_diff = diags(-1 * np.ones(n), offsets=1) + diags(np.ones(n + 1))

    # Distribute activation
    sax_activations_sub = sax_activations.dot(diags(ax_new_mask, dtype=bool))
    sax_activation_cs = csc_matrix(sax_activations.toarray().cumsum(axis=1))
    sax_dist = sax_activations_sub.astype(int).transpose().dot(sax_activation_cs > 0).dot(sax_diff)

    # Compute standardized precision
    ax_p = sax_dist[ax_new_mask, :].toarray().sum(axis=1) * ax_precision[ax_new_mask]
    ax_p -= (sax_dist[ax_new_mask, :].toarray() * ax_precision).sum(axis=1)
    ax_p += (sax_dist.diagonal()[ax_new_mask] * ax_precision[ax_new_mask])
    ax_p /= sax_dist.diagonal()[ax_new_mask]

    return ax_p


def disclose_patterns_multi_output(
        l_completes, server, batch_size, firing_graph, drainer_params, ax_weights, min_firing,
        n_overlap, min_precision, max_precision, min_gain, max_candidate, mapping_feature_input
):
    """

    :param server:
    :param firing_graph:
    :param batch_size:
    :param drainer_params:
    :param ax_weights:
    :param min_firing:
    :param overlap:
    :param min_precision:
    :param max_precision:
    :return:
    """
    l_partials, l_new_completes, n = [], [], 0
    sax_i = server.next_forward(batch_size, update_step=False).sax_data_forward
    for i in range(server.n_label):

        # Get sub partition complete pattern
        l_partition_sub = [partition for partition in firing_graph.partitions if partition['label_id'] == i]
        l_completes_sub = [p for p in l_completes if p.label_id == i]
        if len(l_partition_sub) == 0:
            l_new_completes.extend([p.update_outputs(i, server.n_label) for p in l_completes if p.label_id == i])
            continue

        # Specify keyword args for selection
        l_indices = [p['output_id'] for p in l_partition_sub]
        kwargs = {
            'weight': ax_weights[l_indices], 'p': drainer_params['p'][l_indices], 'r': drainer_params['r'][l_indices],
            "min_precision": min_precision, 'max_precision': max_precision, 'min_gain': min_gain,
            "max_candidate": max_candidate, 'label_id': i
        }
        l_partials_, l_completes_ = disclose_patterns(
            sax_i, l_completes_sub, l_partition_sub, firing_graph, n_overlap, min_firing, mapping_feature_input,
            **kwargs
        )

        # Add extend list of complete and partial patterns
        l_partials.extend(l_partials_)
        l_new_completes.extend([p.update_outputs(i, server.n_label) for p in l_completes_])

    # De-multiply output of partial pred patterns and set server backward pattern
    l_new = [p.update_outputs(i, n_outputs=len(l_partials)) for i, p in enumerate(l_partials)]
    server.pattern_backward = YalaOutputSimplePattern.from_mapping(
        {k: [p.output_id for p in l_new if p.label_id == k] for k in range(server.n_label)}
    )

    return l_new, l_new_completes


def disclose_patterns(
        sax_X, l_selected, l_partitions, firing_graph, n_overlap, min_firing, mapping_feature_input,
        **kwargs
):
    """

    :param X:
    :param l_partitions:
    :param firing_graph:
    :param n_overlap:
    :param min_firing:
    :param drainer_params:
    :param weight:
    :param precision:
    :return:
    """

    # Gather every candidate pattern and compute their activation
    candidate_pattern = get_candidate_pred(
        l_selected, l_partitions, firing_graph, min_firing, mapping_feature_input, **kwargs
    )

    if candidate_pattern is None:
        return [], []

    sax_candidate = candidate_pattern.propagate(sax_X)

    # Set variables for selection
    sax_selected = lil_matrix(sax_candidate.shape)
    ax_is_distinct = np.ones(sax_candidate.shape[1], dtype=bool)
    ax_is_selected = np.zeros(sax_candidate.shape[1], dtype=bool)

    n, l_patterns = 0, []
    for d_score in sorted(candidate_pattern.partitions, key=lambda x: x['precision'], reverse=True):
        if not ax_is_distinct[d_score['output_id']]:
            continue

        # if target precision of a base pattern is not reached, drop the pattern
        if not d_score.get('is_new', True):
            if d_score['precision'] < kwargs['min_precision']:
                continue

        # Update variables
        sax_selected[:, n] = sax_candidate[:, d_score['output_id']] > 0
        ax_is_selected[n] = (not d_score.get('is_new', True) or d_score['precision'] > kwargs['max_precision'])
        ax_is_distinct = update_overlap_mask(sax_selected, sax_candidate, n_overlap)

        # Change index of output and add pattern
        l_patterns.append(YalaBasePattern.from_partition(d_score, candidate_pattern, label_id=kwargs['label_id']))
        n += 1

    # Compute normalized precision
    # ax_norm_precision = get_normalized_precision(
    #     sax_selected[:, :n].tocsc(), np.array([pat.precision for pat in l_patterns]),
    #     ax_is_selected[:n]
    # )

    #
    l_new = [p for i, p in enumerate(l_patterns) if not ax_is_selected[i]]
    l_selected = [pat for i, pat in enumerate(l_patterns) if ax_is_selected[i]]
    #l_selected = [p for i, p in enumerate(l_selected) if ax_norm_precision[i] > kwargs['min_precision']]

    return l_new, l_selected


def get_candidate_pred(l_selected, l_partitions, firing_graph, min_firing, mapping_feature_input, **kwargs):
    """

    :param l_partitions:
    :param firing_graph:
    :param min_firing:
    :param kwargs:
    :return:
    """
    # Get indices of transient and pred patterns
    index_trans = 1 if len(l_partitions[0]['indices']) == 3 else 0
    ax_trans_indices = np.hstack([p['indices'][index_trans] for p in l_partitions])

    # Extract base patterns if any
    if index_trans == 1:

        l_partitions_base = [{k: p[k] if k != 'indices' else [p['indices'][0]] for k in p.keys()} for p in l_partitions]
        base_pattern = YalaBasePatterns.from_partitions(
            l_partitions_base, firing_graph, 'isolate', *[('is_new', False)]
        )\
            .augment_from_patterns(l_selected, 'isolate', *[('is_new', False)])
    else:
        base_pattern = None

    # Extract input matrices and backward fire count of transient patterns
    sax_weight = firing_graph.Iw[:, ax_trans_indices]
    sax_count = firing_graph.backward_firing['i'][:, ax_trans_indices]
    ax_target_precisions = np.array([p['precision'] for p in l_partitions]) + kwargs['min_gain']

    # Compute precision of transient bits organize as its input matrix
    sax_trans_features, sax_trans_input = get_precision(
        sax_weight.astype(float), sax_count.astype(float), kwargs['p'], kwargs['r'], kwargs['weight'], min_firing,
        ax_target_precisions, mapping_feature_input
    )

    # Augment Base patterns instead of passing sax_I, l_precisions, l_levels mother fucker
    return build_pattern(base_pattern, sax_trans_features, sax_trans_input, mapping_feature_input)


def build_pattern(base_pattern, sax_trans_features, sax_trans_input, fi_map):
    """

    """
    # build candidate input matrix
    l_inputs, l_levels, l_partitions, n_cand, n_base = [], [], [], 0, 0
    for i in range(sax_trans_features.shape[1]):
        sax_cand_features = sax_trans_features[:, i]
        sax_cand_input = sax_trans_input[:, i]

        # Get each non zero entry in a single columns
        sax_split_cand_features = diags(sax_cand_features.A.ravel(), format='csc')[:, sax_cand_features.nonzero()[0]]
        sax_split_cand_input = fi_map.dot(sax_split_cand_features > 0).astype(bool)

        # Reduce input of selected features
        sax_split_cand_input = sax_split_cand_input.multiply(sax_cand_input[:, [0] * sax_split_cand_input.shape[1]])

        if sax_split_cand_features.nnz == 0 or sax_split_cand_input.nnz == 0:
            n_base += 1
            continue

        # Append list of precision
        l_partitions.extend(
            [{"precision": p, 'score': None, "output_id": n_cand + n, 'indices': [n_cand + n], "label_id": None}
             for n, p in enumerate(sax_split_cand_features.sum(axis=0).A[0])]
        )
        ax_levels = np.ones(sax_split_cand_input.shape[1], dtype=int)
        n_cand += sax_split_cand_features.shape[1]

        if base_pattern is not None:

            sax_base = base_pattern.I[:, [n_base] * sax_split_cand_input.shape[1]]

            # Get candidates with input that overlap with base
            ax_overlap = sax_split_cand_input.transpose().dot(base_pattern.I[:, n_base]).A.ravel()
            sax_cand_input_mask = fi_map.dot(sax_split_cand_features > 0)

            # Find a correct descriptive comment about operations below
            sax_I = ((sax_base - sax_cand_input_mask.astype(int) > 0) + sax_split_cand_input).multiply(sax_base)
            sax_split_cand_input = sax_I + sax_split_cand_input.dot(diags(~ax_overlap, dtype=bool, format='csc'))

            # Increment levels if necessary
            ax_levels += base_pattern.levels[[n_base] * ax_levels.shape[0]] - ax_overlap

            # Remove old base pattern
            base_pattern = base_pattern.remove(n_base, 'isolate', *[('is_new', False)])

        # Add candidates input and remove corresponding base pattern
        l_inputs.append(sax_split_cand_input.copy())
        l_levels.extend(list(ax_levels))

    if base_pattern is None:
        if l_inputs:
            return YalaBasePatterns.from_input_matrix(hstack(l_inputs), l_partitions, np.array(l_levels))
        else:
            return None

    if l_inputs:
        return base_pattern.augment_from_inputs(hstack(l_inputs), l_partitions, np.array(l_levels), 'isolate')
    else:
        return base_pattern


def get_precision(sax_weight, sax_count, ax_p, ax_r, ax_w, n0, ax_prec, mapping_feature_input):
    """

    :param drainer_params:
    :param score:
    :param t:
    :return:
    """
    # Create mask candidate
    sax_mask = (sax_weight > 0)
    sax_regul = mapping_feature_input.transpose().astype(int).dot(sax_mask)
    sax_weight_r = mapping_feature_input.transpose().dot(sax_weight.multiply(sax_mask))
    sax_count_r = mapping_feature_input.transpose().dot(sax_count.multiply(sax_mask))
    sax_mask_r = (sax_weight_r > 0).multiply(sax_count_r >= n0)

    # compute precision: (original formula: float(score - weight) / (t * (p + r)) + float(p) / (p + r))
    # TODO: Other solution:  compute prec at input level then select correct prec and then compute freq at
    #  feature level with the selected input.
    sax_precision = (sax_weight_r.multiply(sax_mask_r) - (sax_mask_r.multiply(sax_regul).dot(diags(ax_w, format='csc'))))\
        .multiply(sax_mask_r.multiply(sax_count_r.dot(diags(ax_p + ax_r, format='csc'))).power(-1))
    sax_precision += (sax_precision > 0).dot(diags(ax_p / (ax_p + ax_r), format='csc'))

    # Get only precision mask that are larger than target precision (ax_prec)
    precision_mask = sax_precision > (sax_precision > 0).dot(diags(ax_prec, format='csc'))

    return sax_precision.multiply(precision_mask), sax_mask.multiply(mapping_feature_input.dot(precision_mask))


def update_overlap_mask(sax_base, sax_patterns, overlap):
    """

    :param sax_base:
    :param sax_patterns:
    :param overlap_rate:
    :return:
    """
    ax_diff = sax_patterns.astype(int).sum(axis=0) - \
        csc_matrix(sax_base.sum(axis=1)).transpose().astype(int).dot(sax_patterns)
    return np.array(ax_diff)[0] > overlap


def set_feedbacks(ax_phi_old, ax_phi_new, r_max=1000):
    """

    :param phi_old:
    :param phi_new:
    :param r_max:
    :return:
    """
    ax_p, ax_r = np.zeros(ax_phi_new.shape), np.zeros(ax_phi_new.shape)
    for i, (phi_old, phi_new) in enumerate(zip(*[ax_phi_old, ax_phi_new])):
        p, r = set_feedback(phi_old, phi_new, r_max)
        ax_p[i], ax_r[i] = p, r

    return ax_p, ax_r


def set_feedback(phi_old, phi_new, r_max=1000):
    for r in range(r_max):
        p = np.ceil(r * phi_old / (1 - phi_old))
        score = (phi_new * (p + r)) - p
        if score > 0.:
            return p, r

    raise ValueError("Not possible to find feedback values to distinguish {} and {}".format(phi_old, phi_new))