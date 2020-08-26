# Global imports
import numpy as np
from scipy.sparse import csc_matrix, diags, lil_matrix, hstack

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
            # Add Empty base and sampled intersection into a Yala structure
            l_patterns.append(YalaDrainingPattern.from_patterns(
                EmptyPattern(n_inputs, n_outputs, i, i),
                YalaBasePattern.from_input_indices(n_inputs, n_outputs, i, i, sampler.samples[i], 1, ax_weights[i])
            ))

    else:
        for i, pattern in enumerate(sampler.patterns):
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
        n_overlap, min_precision, max_precision, min_gain, max_candidate):
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

        # get partition
        l_partition_sub = [partition for partition in firing_graph.partitions if partition['label_id'] == i]

        if len(l_partition_sub) == 0:
            l_new_completes.extend([p.update_outputs(i, server.n_label) for p in l_completes if p.label_id == i])
            continue

        # Get already selected patterns for the output
        l_completes_sub = [p for p in l_completes if p.label_id == i]

        # Specify keyword args for selection
        l_indices = [p['output_id'] for p in l_partition_sub]
        kwargs = {
            'weight': ax_weights[l_indices], 'p': drainer_params['p'][l_indices], 'r': drainer_params['r'][l_indices],
            "min_precision": min_precision, 'max_precision': max_precision, 'min_gain': min_gain,
            "max_candidate": max_candidate, 'label_id': i
        }
        l_partials_, l_completes_ = disclose_patterns(
            sax_i, l_completes_sub, l_partition_sub, firing_graph, n_overlap, min_firing, **kwargs
        )

        # Add extend list of complete and partial patterns
        l_partials.extend(l_partials_)
        l_new_completes.extend([p.update_outputs(i, server.n_label) for p in l_completes_])

    # De-multiply output of partial pred patterns
    l_new = [p.update_outputs(i, n_outputs=len(l_partials)) for i, p in enumerate(l_partials)]

    # Set server backward pattern
    server.pattern_backward = YalaOutputSimplePattern.from_mapping(
        {k: [p.output_id for p in l_new if p.label_id == k] for k in range(server.n_label)}
    )

    return l_new, l_new_completes


def disclose_patterns(sax_X, l_selected, l_partitions, firing_graph, n_overlap, min_firing, **kwargs):
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
    candidate_pattern = get_candidate_pred(l_partitions, firing_graph, min_firing, **kwargs)

    if candidate_pattern is None:
        return [], []

    # TODO:
    #   * Candidate patterns has partition with precision for old patterns and with (precision, count) for new patternw
    #   * Partition should be again sorted by precision
    #   * Old patterns are considered 2 by 2 diff (because of previous iteration) and different from their 'child'
    #     Thus the only criterion for selection is min proba
    #   * New patterns are selected upon their precision and their overlap with selected other new patterns
    #   * overlap is a function of (precision, count, non_zero_input)
    #  others
    #   * Make sure creating a candidate pattern is usefull or it neither optimal not necessary given new selection
    #     rules
    #   *

    # Set variables for selection
    ax_is_distinct = np.ones(len(candidate_pattern.partitions), dtype=bool)
    ax_is_selected = np.zeros(len(candidate_pattern.partitions), dtype=bool)

    n, l_patterns, l_signature_selected = 0, [], []
    for d_partition in sorted(candidate_pattern.partitions, key=lambda x: x['precision'], reverse=True):
        if not ax_is_distinct[d_partition['output_id']]:
            continue

        # if target precision of a base pattern is not reached, drop the pattern
        if not d_partition.get('is_new', True):
            if d_partition['precision'] < kwargs['min_precision']:
                continue

        # Update variables
        ax_is_selected[n] = (not d_partition.get('is_new', True) or d_partition['precision'] > kwargs['max_precision'])

        # Update distinct array
        if 'signature' in d_partition.keys():
            ax_is_distinct = update_overlap_mask(
                ax_is_distinct, candidate_pattern.partitions, d_partition.pop('signature')
            )

        # Change index of output and add pattern
        l_patterns.append(YalaPredPattern.from_partition(d_partition, candidate_pattern, label_id=kwargs['label_id']))
        n += 1

    # TODO: do it when everything is selected (not here) compute normalized precision
    # ax_norm_precision = get_normalized_precision(
    #     sax_selected[:, :n].tocsc(), np.array([pat.precision for pat in l_patterns]),
    #     ax_is_selected[:n]
    # )

    #
    l_new = [p for i, p in enumerate(l_patterns) if not ax_is_selected[i]]
    l_selected.extend([pat for i, pat in enumerate(l_patterns) if ax_is_selected[i]])
    #l_selected = [p for i, p in enumerate(l_selected) if ax_norm_precision[i] > kwargs['min_precision']]

    return l_new, l_selected


def get_candidate_pred(l_partitions, firing_graph, min_firing, **kwargs):
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
    sax_pred_I = firing_graph.I[:, ax_pred_indices]
    l_pred_precision = list(filter(lambda x: x is not None, [p['partitions'][1]['precision'] for p in l_partitions]))

    # Extract precision and backward fire count of transient patterns
    sax_trans_count = firing_graph.backward_firing['i'][:, ax_trans_indices]
    sax_trans_precision = get_transient_precision(
        firing_graph.Iw[:, ax_trans_indices].astype(float), sax_trans_count.astype(float), kwargs['p'], kwargs['r'],
        kwargs['weight'], min_firing, [p.get('precision', 0) + kwargs['min_gain'] for p in l_partitions]
    )

    return build_pattern(sax_pred_I, l_pred_precision, sax_trans_precision, sax_trans_count)


def build_pattern(sax_pred_I, l_pred_precision, sax_trans_precision, sax_trans_count):
    """

    :param sax_I:
    :param l_prec:
    :param sax_trans:
    :param max_candidate:
    :return:
    """
    # Build candidate input matrix
    l_trans_inputs, l_updated, l_signature_trans = [], [], []
    for i in range(sax_trans_precision.shape[1]):

        #
        sax_trans_precision_sub = sax_trans_precision[:, i]
        sax_trans_count_sub = sax_trans_count[:, i]

        # Get each non zero entry in a single columns
        l_nonzeros = sax_trans_precision_sub.nonzero()[0]
        sax_split_trans_precision = diags(sax_trans_precision_sub.A.ravel(), format='csc')[:, l_nonzeros]
        sax_split_trans_count = diags(sax_trans_count_sub.A.ravel(), format='csc')[:, l_nonzeros]

        if sax_split_trans_precision.nnz == 0:
            continue

        # Append list of signature (precision, count_firing)
        l_signature_trans.extend(list(zip(
            list(sax_split_trans_precision.sum(axis=0).A[0]), list(sax_split_trans_count.sum(axis=0).A[0])
        )))

        # Build input matrix of candidate predictor and add it to list
        sax_trans_I = (sax_split_trans_precision > 0)
        if sax_pred_I.shape[1] > 0:
            sax_trans_I += sax_pred_I[:, np.ones(sax_trans_I.shape[1], dtype=int) * i]

        l_trans_inputs.append(sax_trans_I)
        l_updated.append(i)

    # Remove previous pattern that has been updated TODO: is it usefull ?
    #sax_pred_I = sax_pred_I[:, [i for i in range(sax_pred_I.shape[1]) if i not in l_updated]]
    #l_pred_precision = [p for i, p in enumerate(l_pred_precision) if i not in l_updated]

    # build partition
    l_partitions = [
        {"indices": [i], "precision": p, "output_id": i, "is_new": False} for i, p in enumerate(l_pred_precision)
    ]
    l_partitions.extend([
        {"indices": [len(l_pred_precision) + i], "precision": p, "signature": p * c,
         "output_id": len(l_pred_precision) + i} for i, (p, c) in enumerate(l_signature_trans)
    ])

    return YalaPredPatterns.from_input_matrix(hstack([sax_pred_I] + l_trans_inputs), l_partitions)


def get_transient_precision(sax_weight, sax_count, ax_p, ax_r, ax_w, n0, ax_prec):
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

    # Get only precision mask that are larger than ax_prec TODO: is it usefull ?
    precision_mask = sax_precision > (sax_precision > 0).dot(diags(ax_prec, format='csc'))

    return sax_precision.multiply(precision_mask)


def update_overlap_mask(ax_is_distinct, l_partitions, signature):
    """

    :param l_partitions:
    :param l_signatures:
    :return:
    """
    if signature is None:
        return

    ax_diff = np.array(
        [[p['output_id'], abs(p['signature'] - signature)] for p in l_partitions if 'signature' in p.keys()]
    )
    ax_is_distinct[ax_diff[:, 0].astype(int)] &= ax_diff[:, 1] > np.percentile(ax_diff[:, 1], 5)

    return ax_is_distinct


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