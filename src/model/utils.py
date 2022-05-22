# Global import
import numpy as np
from random import choices
from string import ascii_uppercase
from scipy.signal import convolve2d
from scipy.sparse import csc_matrix

# Local import
from src.model.core.data_models import FgComponents


def init_sample(n, l, server, bitmap, window_length):

    # Get dimensions
    (n_inputs, n_features) = bitmap.bf_map.shape

    # Sample features
    ax_indices = np.random.choice(n_features,  l * n)
    ax_mask = np.zeros((n_features, n), dtype=bool)
    ax_mask[ax_indices, np.array([i // l for i in range(l * n)])] = True

    # Sample inputs and expand it
    ax_sampled = server.get_random_samples(n).A.astype(int).T * bitmap.bf_map.A.dot(ax_mask.astype(int))

    # Use convulation to explore neighbour area and re-mask
    ax_inputs = convolve2d(ax_sampled, np.ones((window_length, 1)), mode='same')
    ax_inputs = ax_inputs * bitmap.bf_map.A.dot(ax_mask.astype(int))

    # Create comp and compute precisions
    comp = FgComponents(
        inputs=csc_matrix(ax_inputs), levels=(ax_inputs.T.dot(bitmap.bf_map.A) > 0).sum(axis=1),
        partitions=[
            {'label_id': 0, 'id': ''.join(choices(ascii_uppercase, k=5)), "stage": "ongoing"} for _ in range(n)
        ],
    )

    return comp