

def prepare_draining_graph():
    pass


def prepare_amplifier_graph():
    pass


def init_parameters(ax_precision):
    """

    :param ax_precision:
    :return:
    """
    # Get scoring process params
    ax_p, ax_r = set_feedbacks(ax_precision + self.picker.min_gain, ax_precision + (2 * self.picker.min_gain))
    drainer_params = DrainerParameters(
        feedbacks=DrainerFeedbacks(penalties=ax_p, rewards=ax_r),
        weights=((ax_p - (ax_precision * (ax_p + ax_r))) * self.picker.min_firing).astype(int) + 1
    )
    return drainer_params