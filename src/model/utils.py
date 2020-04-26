# Global imports
import numpy as np
from scipy.sparse import csc_matrix, diags, lil_matrix, tril, hstack

# Local import
from .patterns import EmptyPattern, YalaBasePattern, YalaPredPattern, YalaDrainingPattern, \
    YalaDrainingPatterns, YalaPredPatterns


def build_firing_graph(sampler, ax_weights, level=1):
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
    if sampler.patterns is None:
        for i in range(sampler.n_outputs):
            # Add Empty base and sampled intersection into a yala structure
            l_patterns.append(YalaDrainingPattern.from_patterns(
                EmptyPattern(sampler.n_inputs, sampler.n_outputs, i),
                YalaBasePattern.from_input_indices(
                    sampler.n_inputs, sampler.n_outputs, i, sampler.samples[i], level, ax_weights[i]
                )
            ))

    else:
        for i, pattern in enumerate(sampler.patterns):
            # Add Empty base and sampled intersection into a yala structure
            l_patterns.append(YalaDrainingPattern.from_patterns(
                pattern,
                YalaBasePattern.from_input_indices(
                    sampler.n_inputs, sampler.n_outputs, pattern.index_output, sampler.samples[i], level,
                    ax_weights[pattern.index_output]
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


def disclose_patterns_multi_output(X, firing_graph, drainer_params, ax_weights, ax_precision, min_firing,
                                   overlap_rate, target_precision):
    """

    :param X:
    :param y:
    :param firing_graph:
    :param drainer_params:
    :param ax_weights:
    :param ax_precision:
    :param min_firing:
    :param overlap_rate:
    :return:
    """
    l_patterns_new, l_patterns_selected = [], []
    for i in range(firing_graph.n_outputs):
        l_partition_sub = [partition for partition in firing_graph.partitions if partition['index_output'] == i]

        # Specify key word args for selection
        kwargs = {
            'precision': ax_precision[i], 'weight': ax_weights[i], 'p': drainer_params['p'][i],
            'target_precision': target_precision, 'r': drainer_params['r'][i],
            'max_patterns': len(firing_graph.I.nonzero()[0])
        }
        l_new, l_selected = disclose_patterns(X, l_partition_sub, firing_graph, overlap_rate, min_firing, **kwargs)
        l_patterns_new.extend(l_new)
        l_patterns_selected.extend(l_selected)

    return l_patterns_new, l_patterns_selected


def disclose_patterns(sax_X, l_partitions, firing_graph, overlap_rate, min_firing, **kwargs):
    """

    :param X:
    :param l_partitions:
    :param firing_graph:
    :param overlap_rate:
    :param min_firing:
    :param drainer_params:
    :param weight:
    :param precision:
    :return:
    """

    # Gather every candidate pattern and compute their activation
    candidate_pred_pattern = get_score_multi_partition(l_partitions, firing_graph, min_firing, **kwargs)

    if candidate_pred_pattern is None:
        return [], []

    sax_activations_all = candidate_pred_pattern.propagate(sax_X)

    # Set variables
    sax_activations_selected = lil_matrix(sax_activations_all.shape)
    ax_overlap_mask = np.ones(sax_activations_all.shape[1], dtype=bool)
    ax_candidate_selection = np.zeros(sax_activations_all.shape[1], dtype=bool)
    n, l_patterns = 0, []
    for d_score in sorted(candidate_pred_pattern.partitions, key=lambda x: x['precision'], reverse=True):
        if not ax_overlap_mask[d_score['index_output']]:
            continue

        # if target precision of a base pattern is not reached, drop the pattern
        if not d_score.get('is_new', True):
            if d_score['precision'] < kwargs['target_precision']:
                continue

        # Update variables
        sax_activations_selected[:, n] = sax_activations_all[:, d_score['index_output']] > 0
        ax_candidate_selection[n] = not d_score.get('is_new', True)
        ax_overlap_mask = update_overlap_mask(sax_activations_selected, sax_activations_all, overlap_rate, min_firing)

        # Change index of output and add pattern
        l_patterns.append(YalaPredPattern.from_partition(
            d_score, candidate_pred_pattern, index_output=l_partitions[0]['index_output']
        ).update_outputs(0, n_outputs=1))
        n += 1

    # Compute normalized precision
    ax_norm_precision = get_normalized_precision(
        sax_activations_selected[:, :n].tocsc(),
        np.array([pat.precision for pat in l_patterns]),
        ax_candidate_selection[:n]
    )

    l_new = [p for i, p in enumerate(l_patterns) if not ax_candidate_selection[i]]
    l_selected = [pat for i, pat in enumerate(l_patterns) if ax_candidate_selection[i]]
    l_selected = [p for i, p in enumerate(l_selected) if ax_norm_precision[i] > kwargs['target_precision']]

    return l_new, l_selected


def get_score_multi_partition(l_partitions, firing_graph, min_firing, **kwargs):
    """

    :param l_partitions:
    :param firing_graph:
    :param min_firing:
    :param kwargs:
    :return:
    """

    # Get indices of transient and pred patterns
    ax_trans_indices = np.hstack([np.array(p['indices'])[p['partitions'][0]['indices']] for p in l_partitions])
    ax_pred_indices = np.hstack([np.array(p['indices'])[p['partitions'][1]['indices']] for p in l_partitions])

    # Extract input matrices and precision of pred patterns
    sax_pred_input = firing_graph.I[:, ax_pred_indices]
    l_precisions = list(filter(lambda x: x is not None, [p['partitions'][1]['precision'] for p in l_partitions]))

    # Extract input matrices and backward fire count of transient patterns
    sax_weight = firing_graph.Iw[:, ax_trans_indices]
    sax_count = firing_graph.backward_firing['i'][:, ax_trans_indices.ravel()]

    # Compute precision of transient bits
    sax_precision = get_precision(
        sax_weight.astype(float), sax_count.astype(float), kwargs['p'], kwargs['r'], kwargs['weight'], min_firing
    )

    return build_candidate_patterns(sax_pred_input, l_precisions, sax_precision, kwargs['precision'])


def build_candidate_patterns(sax_pred_input, l_precs, sax_transient_prec, target_prec):
    """

    :param sax_pred_I:
    :param l_precs:
    :param sax_transient_prec:
    :return:
    """

    # build candidate input matrix
    l_inputs, n_pred = [sax_pred_input], len(l_precs)
    for i in range(sax_transient_prec.shape[1]):

        # Set target precision of transient
        if len(l_precs):
            target_prec = round(max(target_prec, l_precs[i]), 2) + 5e-3
        else:
            target_prec = round(target_prec, 2) + 5e-3

        # Get each non zero entry in a single columns
        sax_splitted_precision = diags(sax_transient_prec[:, i].A.ravel())
        sax_splitted_precision = sax_splitted_precision.tocsc()[:, (sax_transient_prec[:, i] > target_prec).A.ravel()]

        # Append list of precision
        l_precs.extend(list(sax_splitted_precision.sum(axis=0).A[0]))

        # Build input matrix of candidate predicitor and add it to list
        sax_I = (sax_splitted_precision > 0)
        if sax_pred_input.shape[1] > 0:
            sax_I += sax_pred_input[:, np.ones(sax_splitted_precision.shape[1], dtype=int) * i]
        l_inputs.append(sax_I)

    # build partition
    l_partitions = [
        {"indices": [i], "precision": p, "index_output": i, "is_new": False} for i, p in enumerate(l_precs[:n_pred])
    ]

    l_partitions.extend([
        {"indices": [n_pred + i], "precision": p, "index_output": n_pred + i} for i, p in enumerate(l_precs[n_pred:])
    ])

    return YalaPredPatterns.from_input_matrix(hstack(l_inputs), l_partitions)


def get_precision(sax_weight, sax_count, p, r, w0, n0):
    """

    :param drainer_params:
    :param score:
    :param t:
    :return:
    """
    sax_mask = (sax_weight > 0).multiply(sax_count >= n0)
    sax_precision = (sax_weight.multiply(sax_mask) - (w0 * sax_mask))\
        .multiply(sax_mask.multiply(sax_count * (p + r)).power(-1))
    sax_precision += (sax_precision > 0) * (float(p) / (p + r))

    return sax_precision


def update_overlap_mask(sax_base, sax_patterns, overlap_rate, min_firing):
    """

    :param sax_base:
    :param sax_patterns:
    :param overlap_rate:
    :return:
    """
    ax_diff = sax_patterns.astype(int).sum(axis=0) - \
        csc_matrix(sax_base.sum(axis=1)).transpose().astype(int).dot(sax_patterns)
    return np.array(ax_diff)[0] > (1 - overlap_rate) * min_firing


def set_score_params(ax_phi_old, ax_phi_new, r_max=1000):
    """

    :param phi_old:
    :param phi_new:
    :param r_max:
    :return:
    """
    ax_p, ax_r = np.zeros(ax_phi_new.shape), np.zeros(ax_phi_new.shape)
    for i, (phi_old, phi_new) in enumerate(zip(*[ax_phi_old, ax_phi_new])):
        for r in range(r_max):
            p = np.ceil(r * phi_old / (1 - phi_old))
            score = (phi_new * (p + r)) - p
            if score > 0.:
                ax_p[i], ax_r[i] = p, r
                break

    return ax_p, ax_r