# Global import
from scipy.stats import norm, entropy
from scipy.sparse import lil_matrix, diags
import numpy as np
import matplotlib.pyplot as plt

# Local import
from src.model.patterns import YalaBasePatterns
from .utils import set_feedbacks
from .data_models import DrainerFeedbacks, DrainerParameters


def compute_element_amplifier(ax_inner_sub, ax_origin_mask, ax_base_activations, debug=False):

    n_other, n_origin = ax_inner_sub[0, ax_origin_mask].sum(), ax_inner_sub[0, ~ax_origin_mask].sum()

    ax_base_dist = (ax_base_activations * ~ax_origin_mask / ax_base_activations.sum())
    ax_base_dist *= ax_inner_sub[1, ax_origin_mask].sum() / (n_other + n_origin)

    # validation of origin bits
    ax_origin_bit_dist = ax_inner_sub[0, :] * ~ax_origin_mask / (n_other + n_origin)
    ax_origin_noise_dist = get_binomial_upper_ci(ax_base_dist, 0.9, n_other + n_origin)
    ax_origin_selection = ax_origin_bit_dist > ax_origin_noise_dist

    # Validation of other bits
    ax_base_dist = ax_base_activations * ax_origin_mask / (ax_base_activations * ax_origin_mask).sum()
    ax_other_bit_dist = (ax_inner_sub[0, :] * ax_origin_mask) / n_other
    ax_other_noise_dist = get_binomial_upper_ci(ax_base_dist, 0.85, n_other)
    ax_other_selection = ax_other_bit_dist > ax_other_noise_dist

    # Compute criterions
    d_criterion = {}
    d_criterion['n_origin_selected'] = ax_inner_sub[0, ax_origin_selection].sum()
    d_criterion['n_other_selected'] = ax_inner_sub[0, ax_other_selection].sum()
    d_criterion['n_all_significant'] = d_criterion['n_origin_selected'] + d_criterion['n_other_selected']
    d_criterion['n_vertex'] = n_other + n_origin
    d_criterion['final_criterion'] = d_criterion['n_all_significant'] / d_criterion['n_vertex']

    if debug:
        import IPython
        IPython.embed()

    d_origin_signals = {"bit_dist": ax_origin_bit_dist, "noise_dist": ax_origin_noise_dist, "select": ax_origin_selection}
    d_other_signals = {"bit_dist": ax_other_bit_dist, "noise_dist": ax_other_noise_dist, "select": ax_other_selection}
    return d_criterion, d_origin_signals, d_other_signals


def amplify_debug_display(d_criterion, d_origin_signals, d_other_signals, n):

    print(f'criterion: {d_criterion}')
    fig, l_axes = plt.subplots(1, 3)

    # Plot details of origin bits
    l_axes[0].plot(d_origin_signals['bit_dist'], color="k")
    l_axes[0].plot(d_origin_signals['noise_dist'], '--', color="k")
    l_axes[0].plot(d_origin_signals['select'] * d_origin_signals['noise_dist'], 'o', color="k")
    l_axes[0].set_title(f'Origin dist {n} - amplifier')

    # Plot details of other bits
    l_axes[1].plot(d_other_signals['bit_dist'], color="k")
    l_axes[1].plot(d_other_signals['noise_dist'], '--', color="k")
    l_axes[1].plot(d_other_signals['select'] * d_other_signals['noise_dist'], 'o', color="k")
    l_axes[1].set_title(f'Other dist {n} - amplifier')

    # Plot details of all selcted bits
    l_axes[2].plot((d_other_signals['select'] + d_origin_signals['select']), color="k")
    l_axes[2].set_title(f'dist {n} of selected bits - amplifier')
    plt.show()


def amplify_bits(
        sax_inner, ax_inputs, ax_base_activations, init_level, map_fi, new_select_thresh=0.5, max_select_tresh=0.9,
        debug=False
):

    # Set threshold for already selected bits
    tresh = min(max((float(init_level)) / ax_inputs.T.dot(map_fi.A).sum(), new_select_thresh), max_select_tresh)
    sax_I, level = lil_matrix((len(ax_inputs), 1), dtype=int), 0

    for j in range(map_fi.shape[1]):
        ax_inner_sub, ax_origin_mask = sax_inner.A[:, map_fi.A[:, j]], ~ax_inputs[map_fi.A[:, j]]

        from_parent = (~ax_origin_mask).any()

        d_criterion, d_origin_signals, d_other_signals = compute_element_amplifier(
            ax_inner_sub, ax_origin_mask, ax_base_activations[map_fi.A[:, j]]
        )
        print(f'criterion: {d_criterion}')
        if debug:
            amplify_debug_display(d_criterion, d_origin_signals, d_other_signals, j)

        if (d_criterion['final_criterion'] > tresh) and from_parent:
            sax_I[map_fi.A[:, j], 0] = (d_other_signals['select'] + d_origin_signals['select'])\
                .astype(int)
            level += d_criterion['final_criterion']

        elif (d_criterion['final_criterion'] > new_select_thresh) and not from_parent:
            sax_I[map_fi.A[:, j], 0] = (d_other_signals['select'] + d_origin_signals['select'])\
                .astype(int)

    return sax_I, int(level)


def get_binomial_upper_ci(ax_p, conf, n):
    alpha = 1 - conf
    return ax_p + (norm.ppf(1 - (alpha / 2)) * np.sqrt(ax_p * (1 - ax_p) / n))


def get_amplifier_firing_graph(sax_I, level):

    l_partitions = [
        {'indices': 0, 'output_id': 0, 'label': 0, 'precision': 0},
        {'indices': 1, 'output_id': 1, 'label': 0, 'precision': 0}
    ]
    amplifier = YalaBasePatterns.from_input_matrix(
        sax_I[:, [0, 0]], l_partitions, np.array([level, level - 1])
    )

    return amplifier


def get_drainer_firing_graph(sax_I, level):
    firing_graph = YalaBasePatterns.from_input_matrix(
        sax_I, [{'indices': 0, 'output_id': 0, 'label': 0, 'precision': 0}], np.array([level])
    )
    firing_graph.matrices['Im'] = sax_I

    return firing_graph


def split_drained_graph(sax_weight, sax_count, ax_p, ax_r, ax_w, map_fi, debug=False, save=False):
    # Get input weights and count
    sax_mask = (sax_weight > 0).multiply(sax_count > 0)

    if debug:
        # Compute precision
        sax_nom = sax_weight.multiply(sax_mask) - sax_mask.dot(diags(ax_w, format='csc'))
        sax_denom = sax_mask.multiply(sax_count.dot(diags(ax_p + ax_r, format='csc')))
        sax_precision = sax_nom.multiply(sax_denom.astype(float).power(-1))
        sax_precision += (sax_precision != 0).dot(diags(ax_p / (ax_p + ax_r), format='csc'))

        plot_path = 'DATA/test_new_paradigm/{}'
        for i in range(map_fi.shape[1]):
            ax_precision = sax_precision[map_fi[:, i]].A[0]
            plt.plot(ax_precision, color='b')
            plt.plot((ax_precision > 0).astype(int), color='r')
            plt.title(f'feature {i} - drainer')
            if save:
                plt.savefig(plot_path.format(f'{i}_drainer.png'), format='png')
                plt.clf()
            else:
                plt.show()

    sax_residual_mask = ((sax_count > 0) - sax_mask)
    return sax_mask, sax_residual_mask


def init_param_new(ax_precision, min_gain, min_firing=250):
    ax_p, ax_r = set_feedbacks(ax_precision - (2 * min_gain), ax_precision)
    drainer_params = DrainerParameters(
        feedbacks=DrainerFeedbacks(penalties=ax_p, rewards=ax_r),
        weights=((ax_p - ((ax_precision - 2 * min_gain) * (ax_p + ax_r))) * min_firing).astype(int) + 1
    )
    return drainer_params