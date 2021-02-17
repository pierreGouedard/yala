# Global import
from scipy.stats import norm
from scipy.sparse import lil_matrix, diags
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import shift

# Local import
from src.model.patterns import YalaBasePatterns
from .utils import set_feedbacks
from .data_models import DrainerFeedbacks, DrainerParameters


def compute_element_amplifier(ax_inner_sub, ax_origin_mask, ax_base_activations, ci_select=0.9, debug=False):

    n_other, n_origin = ax_inner_sub[0, ax_origin_mask].sum(), ax_inner_sub[0, ~ax_origin_mask].sum()

    ax_base_dist = (ax_base_activations * ~ax_origin_mask / ax_base_activations.sum())
    ax_base_dist *= ax_inner_sub[1, ax_origin_mask].sum() / (n_other + n_origin)

    # validation of origin bits
    ax_origin_bit_dist = ax_inner_sub[0, :] * ~ax_origin_mask / (n_other + n_origin)
    ax_origin_noise_dist = get_binomial_upper_ci(ax_base_dist, ci_select, n_other + n_origin)
    ax_origin_selection = ax_origin_bit_dist > ax_origin_noise_dist

    # Validation of other bits
    ax_base_dist = ax_base_activations * ax_origin_mask / (ax_base_activations * ax_origin_mask).sum()
    ax_other_bit_dist = (ax_inner_sub[0, :] * ax_origin_mask) / n_other
    ax_other_noise_dist = get_binomial_upper_ci(ax_base_dist, ci_select - 0.05, n_other)
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


def final_bit_selection(fg, map_fi, X, ax_base_activations, noise_level=1):

    n_features = fg.I.T.dot(map_fi).sum() / 2
    fg.levels = np.array([n_features, n_features - noise_level])
    sax_x = fg.propagate(X).tocsc()
    sax_inner = sax_x.astype(int).T.dot(X)

    sax_I, level, debug = lil_matrix((fg.I.shape[0], 1), dtype=int), 0, False
    for j in range(map_fi.shape[1]):
        ax_inner_sub = sax_inner.A[:, map_fi.A[:, j]]
        ax_origin_mask = ~fg.I.A[:, 0][map_fi.A[:, j]]
        ax_base_x_sub = ax_base_activations[map_fi.A[:, j]]

        n_origin = ax_inner_sub[0, ~ax_origin_mask].sum()

        ax_base_dist = (ax_base_x_sub * ~ax_origin_mask / ax_base_x_sub.sum())
        ax_base_dist *= ax_inner_sub[1, ax_origin_mask].sum() / n_origin

        # validation of origin bits
        ax_origin_bit_dist = ax_inner_sub[0, :] * ~ax_origin_mask / n_origin
        ax_origin_noise_dist = get_binomial_upper_ci(ax_base_dist, 0.90, n_origin)
        ax_select = (ax_origin_bit_dist > ax_origin_noise_dist).astype(int)

        # Smooth selection of bits
        stop, ax_select_old = False, ax_select
        while not stop:
            ax_select_new = (ax_select + shift(ax_select, 1, cval=1) + shift(ax_select, -1, cval=1) > 1)
            stop = (ax_select_new == ax_select_old).all()
            ax_select_old = ax_select_new
            ax_select *= ax_select_new

        criterion = ax_inner_sub[0, ax_select_new.astype(bool)].sum() / n_origin
        if criterion > 0.8:
            sax_I[map_fi.A[:, j], 0] = ax_select_new.astype(int)
            level += 1

    final_fg = YalaBasePatterns.from_input_matrix(
        sax_I, [{'indices': 0, 'output_id': 0, 'label': 0, 'precision': 0}], np.array([level])
    )

    return final_fg


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


def create_random_fg(fg, map_fi, level):

    ax_bit_counts = fg.I[:, 0].T.dot(map_fi.astype(int)).A[0]
    test_I = lil_matrix(fg.I[:, 0].shape)

    for j in range(map_fi.shape[1]):
        n_bits = map_fi[:, j].sum()
        ax_rvalues = np.random.binomial(1, ax_bit_counts[j] / n_bits, n_bits)
        test_I[map_fi.A[:, j], 0] = ax_rvalues

    return YalaBasePatterns.from_input_matrix(
        test_I.tocsc(), [{'indices': 0, 'output_id': 0, 'label': 0, 'precision': 0}], np.array([level])
    )
