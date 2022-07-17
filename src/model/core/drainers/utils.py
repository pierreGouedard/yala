# Global import
import numpy as np

# Local import
from src.model.utils.data_models import DrainerFeedbacks


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
