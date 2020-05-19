# Global imports
import numpy as np
from scipy.sparse import csc_matrix, diags, lil_matrix, tril, hstack
from sklearn.model_selection import train_test_split

# Local import
from .patterns import EmptyPattern, YalaBasePattern, YalaPredPattern, YalaDrainingPattern, \
    YalaDrainingPatterns, YalaPredPatterns, YalaOutputSimplePattern


def build_firing_graph(sampler, ax_weights, n_inputs=None, n_outputs=None):
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
        for i in range(n_outputs):
            # Add Empty base and sampled intersection into a yala structure
            l_patterns.append(YalaDrainingPattern.from_patterns(
                EmptyPattern(n_inputs, n_outputs, i),
                YalaBasePattern.from_input_indices(n_inputs, n_outputs, i, sampler.samples[i], 1, ax_weights[i])
            ))

    else:
        for i, pattern in enumerate(sampler.patterns):
            # Add Empty base and sampled intersection into a yala structure
            l_patterns.append(YalaDrainingPattern.from_patterns(
                pattern,
                YalaBasePattern.from_input_indices(
                    pattern.n_inputs, pattern.n_outputs, pattern.index_output, sampler.samples[i], 1,
                    ax_weights[pattern.index_output]
                )
            ))

    # Replace by YalaDrainingPatterns.from_patterns
    firing_graph = YalaDrainingPatterns.from_patterns(l_patterns)

    return firing_graph


def select_patterns(l_selected, l_dropouts, firing_graph, X, dropout_rate=0.2):

    l_selected_old = []
    if firing_graph is not None:
        l_selected_old = [YalaPredPattern.from_partition(partition, firing_graph)
                          for partition in firing_graph.partitions]

    # Merge firing graph
    if dropout_rate > 0:

        # Dropout patterns
        l_candidates = l_selected + l_dropouts + l_selected_old
        l_dropout_indices = [i for i in range(len(l_candidates)) if np.random.binomial(1, dropout_rate)]

        # Build lists
        l_dropouts_new = [c for i, c in enumerate(l_candidates) if i in l_dropout_indices]
        l_selected_new = [c for i, c in enumerate(l_candidates) if i not in l_dropout_indices]

    else:
        l_selected_new, l_dropouts_new = l_selected + l_dropouts + l_selected_old, []

    print(len(l_dropouts_new))
    print(len(l_selected_new))
    firing_graph = YalaPredPatterns.from_pred_patterns(l_base_patterns=l_selected_new)

    return firing_graph, l_dropouts_new


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


def disclose_patterns_multi_output(l_selected, server, batch_size, firing_graph, drainer_params, ax_weights, min_firing,
                                   overlap_rate, min_precision, max_precision):
    """

    :param server:
    :param firing_graph:
    :param batch_size:
    :param drainer_params:
    :param ax_weights:
    :param min_firing:
    :param overlap_rate:
    :param min_precision:
    :param max_precision:
    :return:
    """
    d_new, d_selected, n = {}, {}, 0
    sax_i = server.next_forward(batch_size, update_step=False).sax_data_forward
    for i, v in server.get_outputs().items():

        # get partition
        l_partition_sub = [partition for partition in firing_graph.partitions if partition['index_output'] in v]
        l_partition_sub = sorted(l_partition_sub, key=lambda p: p['index_output'])

        # Get already selected patterns for the output
        l_selected_sub = [p for p in l_selected if p.index_output == i]

        # Specify keyword args for selection
        kwargs = {
            'weight': ax_weights[v], 'p': drainer_params['p'][v], 'r': drainer_params['r'][v],
            "min_precision": min_precision, 'max_precision': max_precision
        }
        l_new, l_selected_ = disclose_patterns(
            sax_i, l_selected_sub, l_partition_sub, firing_graph, overlap_rate, min_firing, **kwargs
        )

        # Update output mapping
        d_new[i] = list(zip(*(list(range(n, n + len(l_new))), l_new)))
        n += len(l_new)

        # Extend list of new patterns
        d_selected[i] = l_selected_

    # Get list of selected and new patterns
    l_selected = [p.update_outputs(i, len(d_selected)) for i, l_p in d_selected.items() for p in l_p]
    l_new = [p.update_outputs(i, n_outputs=n) for l_pats in d_new.values() for i, p in l_pats]

    # set server backward pattern
    server.pattern_backward = YalaOutputSimplePattern.from_mapping(
        {k: [i for i, _ in l_pats] for k, l_pats in d_new.items()}
    )

    return sorted(l_new, key=lambda p: p.index_output), l_selected


def disclose_patterns(sax_X, l_selected, l_partitions, firing_graph, overlap_rate, min_firing, **kwargs):
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
    candidate_pattern = get_candidate_pred(l_selected, l_partitions, firing_graph, min_firing, **kwargs)

    if candidate_pattern is None:
        return [], []

    sax_candidate = candidate_pattern.propagate(sax_X)

    # Set variables for selection
    sax_selected = lil_matrix(sax_candidate.shape)
    ax_is_distinct = np.ones(sax_candidate.shape[1], dtype=bool)
    ax_is_selected = np.zeros(sax_candidate.shape[1], dtype=bool)

    n, l_patterns = 0, []
    for d_score in sorted(candidate_pattern.partitions, key=lambda x: x['precision'], reverse=True):
        if not ax_is_distinct[d_score['index_output']]:
            continue

        # if target precision of a base pattern is not reached, drop the pattern
        if not d_score.get('is_new', True):
            if d_score['precision'] < kwargs['min_precision']:
                continue

        # Update variables
        sax_selected[:, n] = sax_candidate[:, d_score['index_output']] > 0
        ax_is_selected[n] = (not d_score.get('is_new', True) or d_score['precision'] > kwargs['max_precision'])
        ax_is_distinct = update_overlap_mask(sax_selected, sax_candidate, overlap_rate, min_firing)

        # Change index of output and add pattern
        l_patterns.append(YalaPredPattern.from_partition(
            d_score, candidate_pattern, index_output=l_partitions[0]['index_output']
        ))
        n += 1

    # Compute normalized precision
    ax_norm_precision = get_normalized_precision(
        sax_selected[:, :n].tocsc(), np.array([pat.precision for pat in l_patterns]),
        ax_is_selected[:n]
    )

    #
    l_new = [p for i, p in enumerate(l_patterns) if not ax_is_selected[i]]
    l_selected = [pat for i, pat in enumerate(l_patterns) if ax_is_selected[i]]
    l_selected = [p for i, p in enumerate(l_selected) if ax_norm_precision[i] > kwargs['min_precision']]

    return l_new, l_selected


def get_candidate_pred(l_selected, l_partitions, firing_graph, min_firing, **kwargs):
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

    # Extract activation matrices and precision of pred patterns
    sax_pred = hstack([firing_graph.I[:, ax_pred_indices]] + [p.I[:, 0] for p in l_selected])
    l_precisions = [p['partitions'][1]['precision'] for p in l_partitions] + [p.precision for p in l_selected]
    l_precisions = list(filter(lambda x: x is not None, l_precisions))

    # Extract input matrices and backward fire count of transient patterns
    sax_weight = firing_graph.Iw[:, ax_trans_indices]
    sax_count = firing_graph.backward_firing['i'][:, ax_trans_indices.ravel()]

    # Compute precision of transient bits organize as its input matrix
    sax_trans = get_precision(
        sax_weight.astype(float), sax_count.astype(float), kwargs['p'], kwargs['r'], kwargs['weight'], min_firing
    )

    return build_pattern(sax_pred, l_precisions, sax_trans)


def build_pattern(sax_pred, l_precisions, sax_trans):
    """

    :param sax_pred_I:
    :param l_precs:
    :param sax_transient_prec:
    :return:
    """
    # build candidate input matrix
    l_inputs, n_pred = [sax_pred], len(l_precisions)
    for i in range(sax_trans.shape[1]):

        # Get each non zero entry in a single columns
        sax_split_trans = diags(sax_trans[:, i].A.ravel(), format='csc')[:, sax_trans[:, i].nonzero()[0]]

        if sax_split_trans.nnz == 0:
            continue

        # Append list of precision
        l_precisions.extend(list(sax_split_trans.sum(axis=0).A[0]))

        # Build input matrix of candidate predictor and add it to list
        sax_I = (sax_split_trans > 0)

        if sax_pred.shape[1] > 0:
            sax_I += sax_pred[:, np.ones(sax_split_trans.shape[1], dtype=int) * i]

        l_inputs.append(sax_I)

    # build partition
    l_partitions = [
        {"indices": [i], "precision": p, "index_output": i, "is_new": False} for i, p in enumerate(l_precisions[:n_pred])
    ]

    l_partitions.extend([
        {"indices": [n_pred + i], "precision": p, "index_output": n_pred + i} for i, p in enumerate(l_precisions[n_pred:])
    ])

    return YalaPredPatterns.from_input_matrix(hstack(l_inputs), l_partitions)


def get_precision(sax_weight, sax_count, ax_p, ax_r, ax_w, n0):
    """

    :param drainer_params:
    :param score:
    :param t:
    :return:
    """
    # Create mask candidate
    sax_mask = (sax_weight > 0).multiply(sax_count >= n0)

    # compute precision: (original formula: float(score - weight) / (t * (p + r)) + float(p) / (p + r))
    sax_precision = (sax_weight.multiply(sax_mask) - (sax_mask.dot(diags(ax_w, format='csc'))))\
        .multiply(sax_mask.multiply(sax_count.dot(diags(ax_p + ax_r, format='csc'))).power(-1))
    sax_precision += (sax_precision > 0).dot(diags(ax_p / (ax_p + ax_r), format='csc'))

    return sax_precision.multiply(sax_precision > 0)


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