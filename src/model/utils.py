# Global imports
import numpy as np

# Local import
from .patterns import EmptyPattern, YalaBasePattern, YalaTransientPattern, YalaSingleDrainingPattern, \
    YalaMutlipleDrainingPattern


def build_firing_graph(sampler, ax_weights, return_patterns=False):
    """

    :param sampler:
    :param weights:
    :param return_patterns:
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

    if return_patterns:
        return firing_graph, l_patterns

    return firing_graph


def augment_multi_output_patterns(X, y, firing_graph, drainer_params, ax_weights, min_firing, overlap_rate):

    l_patterns, n_updates = [], 0
    for i in range(firing_graph.n_outputs):
        l_partition_sub = [partition for partition in firing_graph.partitions if partition['index_output'] == i]
        l_sub_patterns, n = augment_patterns(
            X, y, l_partition_sub, firing_graph, overlap_rate, min_firing, drainer_params, ax_weights
        )

        l_patterns.extend(l_sub_patterns)
        n_updates += n

    return l_patterns, n_updates


def augment_patterns(X, y, l_partitions, firing_graph, overlap_rate, min_firing, drainer_params, ax_weights):
    """

    :param X:
    :param y:
    :param l_partitions:
    :param firing_graph:
    :param overlap_rate:
    :param min_firing:
    :param drainer_params:
    :return:
    """

    l_patterns, n, ax_base, ax_y = [], 0, np.zeros(X.shape[0], dtype=bool), y.toarray()[:, 0].astype(int)
    for partition in l_partitions:

        # Init selected pattern list
        l_pattern_sub = []

        # Extract Yala Single draining pattern and base pattern
        drained_pattern, base_pattern, transient_pattern = extract_draining_pattern(partition, firing_graph)

        # Get score for bits of each vertex of transient partition
        d_ind = dict(zip(*[range(transient_pattern.n_intersection), [True] * transient_pattern.n_intersection]))
        d_args = {
            'precision': base_pattern.precision if base_pattern is not None else 0.,
            'weight': ax_weights[transient_pattern.index_output],
            'p': drainer_params['p'][transient_pattern.index_output],
            'r': drainer_params['r'][transient_pattern.index_output]
        }
        l_ind_scores = get_transient_scores(transient_pattern, min_firing, **d_args)

        # Augment patterns with non overlapping best candidate
        for ind, d_bit in l_ind_scores:
            if not d_ind.get(ind, False):
                continue

            # Update pattern
            if base_pattern is not None:
                pattern = base_pattern.copy().augment([d_bit['index']], precision=d_bit['precision'])

            else:
                pattern = YalaBasePattern.from_input_indices(
                    firing_graph.n_inputs, firing_graph.n_outputs, partition['index_output'], [[d_bit['index']]],
                    **{'precision': d_bit['precision']}
                )

            # Validate overlapping rate
            ax_pattern, is_overlapping = overlap_test(X, ax_base, ax_y, pattern, overlap_rate)
            if not is_overlapping:
                ax_base = (ax_base + ax_pattern) > 0
                l_pattern_sub.append(pattern)
                _ = d_ind.pop(ind)

        n += len(l_pattern_sub)

        if len(l_pattern_sub) == 0 and base_pattern is not None:
            l_pattern_sub.append(base_pattern)

        l_patterns.extend(l_pattern_sub)

    return l_patterns, n


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

    return drained_pattern, base_pattern, transient_pattern


def get_transient_scores(transient_pattern, min_firing, **kwargs):
    """

    :param transient_pattern:
    :param min_firing:
    :return:
    """
    l_ind_scores = [
        (i, d_score) for i in range(transient_pattern.n_intersection)
        for d_score in get_bit_scores(transient_pattern, i, min_firing, **kwargs)
    ]

    # Sort vertex of transient according to the maximum precision of their input bits
    l_ind_scores = sorted(l_ind_scores, key=lambda t: t[1]['precision'], reverse=True)

    return l_ind_scores


def get_bit_scores(firing_graph, ind, min_firing, **kwargs):
    """

    :param firing_graph:
    :param ind:
    :param min_firing:
    :return:
    """
    # Get quantity of interest
    sax_scores, sax_t = firing_graph.Iw[:, ind], firing_graph.backward_firing['i'][:, ind]

    if sax_scores.nnz > 0:
        l_indices = [i for i in (sax_scores > 0).nonzero()[0] if sax_t[i, 0] >= min_firing]
        l_precisions = [
            get_precision(kwargs['p'], kwargs['r'], kwargs['weight'], sax_scores[i, 0], sax_t[i, 0]) for i in l_indices
        ]
        for ind, precision in sorted(zip(l_indices, l_precisions), key=lambda t: t[1], reverse=True):
            if precision > kwargs['precision']:
                yield {'index': ind, 'precision': precision}


def get_precision(p, r, weight, score, t):
    """

    :param drainer_params:
    :param score:
    :param t:
    :return:
    """
    return float(score - weight) / (t * (p + r)) + float(p) / (p + r)


def overlap_test(X, ax_base, ax_mask, pattern, overlap_rate):
    """

    :param X:
    :param ax_base:
    :param ax_mask:
    :param pattern:
    :param overlap_rate:
    :return:
    """
    # compute output of pattern
    ax_pattern = pattern.propagate(X).toarray()[:, 0] * ax_mask

    return ax_pattern, ax_base.astype(int).dot(ax_pattern) > overlap_rate * ax_pattern.sum()


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