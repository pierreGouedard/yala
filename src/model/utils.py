# Global import
from scipy.sparse import hstack, lil_matrix
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import convolve2d
from scipy.sparse import csc_matrix

# Local import
from src.model.patterns import YalaBasePatterns, YalaTopPattern
from .data_models import DrainerFeedbacks, DrainerParameters, FgComponents
from scipy.signal import convolve


def init_sample(n, server, l0, sax_bf_map, window_length=3, n_sample_precision=100000):

    # Get dimensions
    (n_inputs, n_features) = sax_bf_map.shape

    # Sample features
    ax_indices = np.random.choice(n_features,  l0 * n)
    ax_mask = np.zeros((n_features, n), dtype=bool)
    ax_mask[ax_indices, np.array([i // l0 for i in range(l0 * n)])] = True

    # Sample inputs and expand it
    ax_inputs = convolve2d(server.get_random_samples(n).A.astype(int).T, np.ones((window_length, 1)), mode='same')
    ax_inputs = ax_inputs * sax_bf_map.A.dot(ax_mask.astype(int))

    # Create comp and compute precisions
    bottom_comp = FgComponents(
        inputs=csc_matrix(ax_inputs), levels=(ax_inputs.T.dot(sax_bf_map.A) > 0).sum(axis=1),
        partitions=[{'label_id': 0} for _ in range(n)]
    )
    bfg = YalaBasePatterns.from_fg_comp(bottom_comp)

    # Get masked activations
    sax_x = server.next_masked_forward(n=n_sample_precision, update_step=False)
    sax_y = server.next_backward(n=n_sample_precision, update_step=False).sax_data_backward

    # Compute mean precision of each vertex and compute precision
    sax_x = bfg.propagate(sax_x)
    ax_precisions = (sax_y.T.astype(int).dot(sax_x) / (sax_x.sum(axis=0) + 1e-6)).A[0]
    import IPython
    IPython.embed()
    # Set drainer params and weight of firing graph
    drainer_params = init_parameters(ax_precisions, min_gain, min_firing)
    bfg.matrices['Iw'] = bfg.I * drainer_params.weights[0]

    # Update backward pattern (1 label only)
    server.pattern_backward = YalaTopPattern.from_mapping({0: list(range(bfg.n_vertex))})

    # TODO: add fucking mask of pattern here
    print(f'precisions: {ax_precisions}')

    # Update partitions with freshly computed precision
    firing_graph.partitions = [{**p, 'precision': ax_precisions[i]} for i, p in enumerate(firing_graph.partitions)]

    # Get precision

    # Get comp


    return


def prepare_refinement(fg_comp, server, n_sample_precision, min_gain, min_firing):

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


def prepare_expansion(fg_comp, server, n_sample_precision, min_gain, min_firing):

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