# Global imports
import numpy as np
from scipy.sparse import csc_matrix, diags, lil_matrix, tril

# Local import
from .patterns import EmptyPattern, YalaBasePattern, YalaTransientPattern, YalaSingleDrainingPattern, \
    YalaMutlipleDrainingPattern, YalaPredictingPattern


def build_firing_graph(sampler, ax_weights):
    """

    :param sampler:
    :param weights:
    :return:
    """

    if sampler.vertices is None:
        raise ValueError(
            "Before Building firing graph, one need to sample input bits using generative or discriminative "
            "sampling"
        )

    l_patterns = []
    if sampler.patterns is None:
        for i in range(sampler.n_outputs):
            # Add Empty base and sampled intersection into a yala structure
            l_patterns.append(YalaSingleDrainingPattern.from_patterns(
                EmptyPattern(sampler.n_inputs, sampler.n_outputs, i),
                YalaTransientPattern.from_input_indices(
                    sampler.n_inputs, sampler.n_outputs, sampler.l0, i, sampler.vertices[i], ax_weights[i]
                )
            ))

    else:
        for i, pattern in enumerate(sampler.patterns):
            # Add Empty base and sampled intersection into a yala structure
            l_patterns.append(YalaSingleDrainingPattern.from_patterns(
                pattern,
                YalaTransientPattern.from_input_indices(
                    sampler.n_inputs, sampler.n_outputs, sampler.l0, pattern.index_output, sampler.vertices[i],
                    ax_weights[pattern.index_output]
                )
            ))

    # Replace by YalaDrainingPatterns.from_patterns
    firing_graph = YalaMutlipleDrainingPattern.from_patterns(l_patterns)

    return firing_graph


def get_normalized_precision(sax_activations, ax_precision, ax_new_mask, overlap_rate=0.):

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

    if overlap_rate > 0:
        sax_temp = (tril(sax_dist, k=-1) * (1 - overlap_rate)).floor()
        sax_dist = sax_dist - sax_temp + diags(np.squeeze(np.asarray(sax_temp.sum(axis=1))))

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
    import time
    t0 = time.time()
    # Get score for every partition
    l_scores = get_scores_multi_partition(l_partitions, firing_graph, min_firing, **kwargs)

    if len(l_scores) == 0:
        return [], []

    # Gather every candidate pattern and compute their activation
    sax_activations_all = YalaPredictingPattern.from_base_patterns([d['pattern'] for d in l_scores]) \
        .propagate(sax_X)
    sax_activations_all = sax_activations_all[:, range(max(sax_activations_all.nonzero()[1]) + 1)]
    print('time get score and propagate {} seconds for {} candidates'.format(time.time() - t0, len(l_scores)))

    # Set variables
    sax_activations_selected = lil_matrix(sax_activations_all.shape)
    ax_overlap_mask = np.ones(sax_activations_all.shape[1], dtype=bool)
    ax_candidate_selection = np.zeros(sax_activations_all.shape[1], dtype=bool)
    n, l_patterns = 0, []
    for d_score in sorted(l_scores, key=lambda d: d['precision'], reverse=True):
        if not ax_overlap_mask[d_score['pattern'].index_output]:
            continue

        # if target precision of a base pattern is not reached, drop the pattern
        if d_score['is_base']:
            if d_score['precision'] < kwargs['target_precision']:
                continue

        # Update variables
        sax_activations_selected[:, n] = sax_activations_all[:, d_score['pattern'].index_output] > 0
        ax_candidate_selection[n] = d_score['is_base']
        ax_overlap_mask = update_overlap_mask(sax_activations_selected, sax_activations_all, overlap_rate, min_firing)

        # change index of output and add pattern
        l_patterns.append(d_score['pattern'].copy().update_outputs(l_partitions[0]['index_output'], 1))
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


def get_scores_multi_partition(l_partitions, firing_graph, min_firing, **kwargs):
    """

    :param l_partitions:
    :param firing_graph:
    :param min_firing:
    :return:
    """
    n, l_scores = 0, []
    for i, partition in enumerate(l_partitions):

        # Extract Yala Single draining pattern and base pattern
        base_pattern, transient_pattern = extract_draining_pattern(partition, firing_graph)

        # Update kwargs for score computation
        if base_pattern is not None:
            kwargs.update({'precision': max(kwargs['precision'], base_pattern.precision)})

        for j in range(transient_pattern.n_intersection):
            l_scores_sub = list(get_scores(transient_pattern, base_pattern, j, i, min_firing, **kwargs))
            kwargs.update({'n': kwargs.get('n', 0) + len(l_scores_sub)})
            l_scores.extend(l_scores_sub)

    return l_scores


def extract_draining_pattern(partition, firing_graph):
    """

    :param partition:
    :param firing_graph:
    :return:
    """
    drained_pattern = YalaSingleDrainingPattern.from_partition(partition, firing_graph, add_backward_firing=True)

    d_partitions = {sub_part['name']: sub_part for sub_part in drained_pattern.partitions}

    base_pattern = YalaBasePattern.from_partition(
        d_partitions['base'], drained_pattern, index_output=partition['index_output']
    )

    transient_pattern = YalaTransientPattern.from_partition(
        d_partitions['transient'], drained_pattern, index_output=partition['index_output'], add_backward_firing=True
    )

    return base_pattern, transient_pattern


def get_scores(transient_pattern, base_pattern, ind, base_id, min_firing, **kwargs):
    """

    :param transient_pattern:
    :param base_pattern:
    :param ind:
    :param min_firing:
    :return:
    """
    # Get quantity of interest
    pattern_args = {'n_patterns': kwargs['max_patterns']}
    if base_pattern is None:
        pattern_args.update({'n_inputs': transient_pattern.n_inputs})

    #
    sax_scores, sax_t = transient_pattern.Iw[:, ind], transient_pattern.backward_firing['i'][:, ind]
    n = kwargs.get('n', 0)

    # For each valid candidate yield the pattern and the precision
    if sax_scores.nnz > 0:
        # TODO: Add a property of transient pattern called "selectable":
        #  A transient pattern is "selectable" iff # of input > level, else it is not selectable, in this case
        #  Compute score of each indices
        l_indices = [i for i in (sax_scores > 0).nonzero()[0] if sax_t[i, 0] >= min_firing]
        l_precisions = [
            get_precision(sax_scores[i, 0], sax_t[i, 0], kwargs['p'], kwargs['r'], kwargs['weight']) for i in l_indices
        ]

        for ind, precision in sorted(zip(l_indices, l_precisions), key=lambda t: t[1], reverse=True):
            if int(precision * 100) > int(kwargs['precision'] * 100):
                yield {
                    'is_base': False,
                    'pattern': build_pattern(ind, base_pattern, precision, n, **pattern_args),
                    'precision': precision,
                    'base_id': base_id
                }
                n += 1

    # Yield the base pattern with its precision
    if base_pattern is not None:
        pattern = base_pattern.copy().update_outputs(n, pattern_args['n_patterns'])
        yield {'is_base': True, 'pattern': pattern, 'precision': pattern.precision, 'base_id': base_id}


def build_pattern(index, base_pattern, precision, index_output, n_patterns, n_inputs=None):
    # Update pattern
    if base_pattern is not None:
        pattern = base_pattern.copy()\
            .update_outputs(index_output, n_patterns)\
            .augment([index], precision=precision)

    else:
        pattern = YalaBasePattern.from_input_indices(
            n_inputs, n_patterns, index_output, [index], **{'precision': precision}
        )

    return pattern


def get_precision(score, t, p, r, weight):
    """

    :param drainer_params:
    :param score:
    :param t:
    :return:
    """
    return float(score - weight) / (t * (p + r)) + float(p) / (p + r)


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