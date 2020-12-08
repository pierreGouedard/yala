# Global imports
import numpy as np
from scipy.sparse import csc_matrix, diags

# Local import
from .patterns import YalaDrainingPatterns


def build_draining_firing_graph(sampler, drainer_params, pattern=None):
    sax_inputs = sampler.sample(pattern).sax_inputs.dot(diags(drainer_params.weights, dtype=float))
    return YalaDrainingPatterns.from_input_matrix(sax_inputs, drainer_params, sampler.n_label, base_patterns=pattern)


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