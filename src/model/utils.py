# Global import
import numpy as np
from random import choices
from string import ascii_uppercase, digits
from scipy.signal import convolve2d
from scipy.sparse import csc_matrix

# Local import
from src.model.helpers.data_models import DrainerFeedbacks, FgComponents


def init_sample(n, server, l0, sax_bf_map, window_length):

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
        partitions=[{'label_id': 0, 'id': ''.join(choices(ascii_uppercase + digits, k=5))} for _ in range(n)],
    )
    return bottom_comp


def init_parameters(drainer_params, min_firing):

    # Get params
    ax_precision, margin = drainer_params.precisions, drainer_params.margin
    ax_precision = ax_precision.clip(max=1., min=margin + 0.01)

    # Compute penalty and reward values
    ax_p, ax_r = set_feedbacks(ax_precision - margin, ax_precision - (margin / 2))
    drainer_params.feedbacks = DrainerFeedbacks(penalties=ax_p, rewards=ax_r)

    # Compute weights
    drainer_params.weights = ((ax_p - ((ax_precision - margin) * (ax_p + ax_r))) * min_firing).astype(int) + 1

    return drainer_params


def set_feedbacks(ax_phi_old, ax_phi_new, r_max=1000):
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
