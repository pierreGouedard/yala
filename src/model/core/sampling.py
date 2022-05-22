# Global import
import numpy as np


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


def sample_from_mask(ax_mask, n=None, ax_n=None):
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