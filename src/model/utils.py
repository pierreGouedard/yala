# Global import
from scipy.sparse import hstack

# Local import
from src.model.patterns import YalaBasePatterns, YalaTopPattern
from .utils import set_feedbacks
from .data_models import DrainerFeedbacks, DrainerParameters, FgComponents


def prepare_draining_graph(fg_comp, server, n_sample_precision, min_gain, min_firing):

    # TODO: does not support multi label in its current state

    # Build firing graph used for draining
    firing_graph = YalaBasePatterns.from_fg_comp(fg_comp)
    firing_graph.matrices['Im'] = firing_graph.I

    # Get masked activations
    sax_x = server.next_masked_forward(n=n_sample_precision, update_step=False)
    sax_y = server.next_backward(n=n_sample_precision, update_step=False).sax_data_backward

    # Compute mean precision of each vertex and compute precision
    sax_x = firing_graph.propagate(sax_x)
    ax_precisions = (sax_y.T.astype(int).dot(sax_x) / (sax_x.sum(axis=0) + 1e-6)).A[0]

    # Set drainer params and weight of firing graph
    drainer_params = init_parameters(ax_precisions, min_gain, min_firing)
    firing_graph.matrices['Iw'] = firing_graph.I * drainer_params.weights[0]

    # Update backward pattern (1 label only)
    server.pattern_backward = YalaTopPattern.from_mapping({0: list(range(firing_graph.n_vertex))})
    print(f'precisions: {ax_precisions}')

    # Update partitions with freshly computed precision
    firing_graph.partitions = [{**p, 'precision': ax_precisions[i]} for i, p in enumerate(firing_graph.partitions)]

    return firing_graph, drainer_params


def prepare_amplifier_graph(drained):

    sax_weight, sax_count, n = drained.Iw, drained.backward_firing['i'], drained.n_vertex

    # Get input weights and count
    sax_left_mask = (sax_weight > 0).multiply((sax_count > 0))
    sax_right_mask = ((sax_count > 0).astype(int) - sax_left_mask.astype(int) > 0)

    firing_graph = YalaBasePatterns.from_fg_comp(FgComponents(
        inputs=hstack([sax_left_mask, sax_right_mask])[:, sum([[i, i + n] for i in range(n)], [])],
        partitions=sum([[p, p] for p in drained.partitions], []),
        levels=drained.levels[sum([[i, i] for i in range(n)], [])]
    ))

    l_pairs = [[2 * i, 2 * i + 1] for i in range(n)]

    return firing_graph, l_pairs


def init_parameters(ax_precision, min_gain, min_firing):
    """

    :param ax_precision:
    :return:
    """
    # Clip upper to max precision and get penalty-reward parameters
    ax_precision = ax_precision.clip(max=1. - 2 * min_gain)
    ax_p, ax_r = set_feedbacks(ax_precision + min_gain, ax_precision + (2 * min_gain))

    # Create drainer params
    drainer_params = DrainerParameters(
        feedbacks=DrainerFeedbacks(penalties=ax_p, rewards=ax_r),
        weights=((ax_p - (ax_precision * (ax_p + ax_r))) * min_firing).astype(int) + 1
    )

    return drainer_params