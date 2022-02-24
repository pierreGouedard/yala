# Global import
import numpy as np
from random import choices
from string import ascii_uppercase
from scipy.signal import convolve2d
from scipy.sparse import csc_matrix

# Local import
from src.model.helpers.data_models import DrainerFeedbacks, FgComponents, ShaperProba


def init_sample(n, l, server, sax_bf_map, window_length, support_proba=0.2):

    # Get dimensions
    (n_inputs, n_features) = sax_bf_map.shape

    # Sample features
    ax_indices = np.random.choice(n_features,  l * n)
    ax_mask = np.zeros((n_features, n), dtype=bool)
    ax_mask[ax_indices, np.array([i // l for i in range(l * n)])] = True

    # Sample inputs and expand it
    ax_inputs = convolve2d(server.get_random_samples(n).A.astype(int).T, np.ones((window_length, 1)), mode='same')
    ax_inputs = ax_inputs * sax_bf_map.A.dot(ax_mask.astype(int))

    # Create comp and compute precisions
    bottom_comp = FgComponents(
        inputs=csc_matrix(ax_inputs), levels=(ax_inputs.T.dot(sax_bf_map.A) > 0).sum(axis=1),
        partitions=[
            {'label_id': 0, 'id': ''.join(choices(ascii_uppercase, k=5)), "stage": "ongoing"} for _ in range(n)
        ],
    )

    # Init Shaper probabilities
    shaper_proba = ShaperProba(dim=ax_mask.shape, support_proba=support_proba).set_probas(ax_mask)
    return bottom_comp, shaper_proba


def sample_from_proba(ax_p, n=None, ax_n=None):
    if ax_n is None:
        ax_n = np.ones(ax_p.shape[0], dtype=int) * n

    # Prepare choice
    (ny, nx), ax_linear = ax_p.shape, np.arange(ax_p.shape[1])
    ax_counts = np.minimum((ax_p > 0).sum(axis=1), ax_n) * ((ax_p > 0).sum(axis=1) >= ax_n)

    def masked_choice(ax_p_, n_):
        return list(np.random.choice(ax_linear, n_, replace=False, p=ax_p_))

    # Random coordinate choice
    l_y_ind = sum([[i] * ax_counts[i] for i in range(ny)], [])
    l_x_ind = sum([masked_choice(ax_p[i, :], ax_counts[i]) for i in range(ny)], [])

    # Create new mask features
    ax_mask_sampled = np.zeros((ny, nx), dtype=bool)
    if l_y_ind:
        ax_mask_sampled[l_y_ind, l_x_ind] = True

    return ax_mask_sampled


def sample_from_mask(ax_mask, n=None, ax_n=None, ax_p=None):
    if ax_n is None:
        ax_n = np.ones(ax_mask.shape[0], dtype=int) * n

    # Prepare choice
    (ny, nx), ax_linear = ax_mask.shape, np.arange(ax_mask.shape[1])
    ax_counts = np.minimum(ax_mask.sum(axis=1), ax_n) * (ax_mask.sum(axis=1) >= ax_n)

    def masked_choice(ax_mask, n):
        return list(np.random.choice(ax_linear[ax_mask], n, replace=False))

    # Random coordinate choice
    l_y_ind = sum([[i] * ax_counts[i] for i in range(ny)], [])
    l_x_ind = sum([masked_choice(ax_mask[i, :], ax_counts[i]) for i in range(ny)], [])

    # Create new mask features
    ax_mask_sampled = np.zeros((ny, nx), dtype=bool)
    if l_y_ind:
        ax_mask_sampled[l_y_ind, l_x_ind] = True

    return ax_mask_sampled


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
