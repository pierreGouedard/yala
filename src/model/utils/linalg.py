# Global import
from scipy.sparse import spmatrix, csc_matrix, hstack as sphstack
import numpy as np
from scipy.signal import convolve2d


def expand(sax_inputs, bitmap, n):
    sax_inputs = csc_matrix(convolve2d((sax_inputs > 0).A, np.ones((2 * n + 1, 1)), mode='same') > 0).multiply(
        bitmap.bitmask(sax_inputs > 0)
    ).astype(int)

    return sax_inputs


def shrink(sax_inputs, bitmap, p_shrink=0.4):
    # Init variables
    ax_reduced_inputs, sax_counts = np.zeros(sax_inputs.shape, dtype=bool), bitmap.b2f(sax_inputs.astype(int))
    ax_lengths = np.unique(sax_counts.data)
    sax_counts.data = ((sax_counts.data * p_shrink) / 2).round().clip(min=1).astype(int)

    # Reduce inputs
    for w in ax_lengths:
        win_len = int(w + (w % 2 == 0))
        ax_win = np.ones((win_len, 1))
        ax_mask = bitmap.bf_map.dot(sax_counts.A.T.astype(int))
        ax_reduced_inputs |= convolve2d(sax_inputs.A, ax_win, mode='same') >= (win_len // 2) + 1 + ax_mask

    return csc_matrix(ax_reduced_inputs)


def connex():
    pass
