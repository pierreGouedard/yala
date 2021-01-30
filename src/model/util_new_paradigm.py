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


def single_select_amplified_bits(
        sax_inner, ax_base_activations, ax_amplified_inputs, map_fi, selection_thresh, debug=False
):

    sax_I, level = lil_matrix((len(ax_amplified_inputs), 1), dtype=int), 0
    for j in range(map_fi.shape[1]):
        # Get inner product between amplified signals and grid's bit of the feature
        ax_inner_sub = sax_inner.A[:, map_fi.A[:, j]]
        ax_origin_mask = ~ax_amplified_inputs[map_fi.A[:, j]]
        d_criterion, d_origin_signals, d_other_signals = compute_element_amplifier(
            ax_inner_sub, ax_origin_mask, ax_base_activations[map_fi.A[:, j]]
        )

        print(f"Criterions: {d_criterion}")
        if debug:
            fig, l_axes = plt.subplots(1, 3)

            # Plot details of origin bits
            l_axes[0].plot(d_origin_signals['bit_dist'], color="k")
            l_axes[0].plot(d_origin_signals['noise_dist'], '--', color="k")
            l_axes[0].plot(d_origin_signals['select'] * d_origin_signals['noise_dist'], 'o', color="k")
            l_axes[0].set_title(f'Origin dist {j} - amplifier')

            l_axes[1].plot(d_other_signals['bit_dist'], color="k")
            l_axes[1].plot(d_other_signals['noise_dist'], '--', color="k")
            l_axes[1].plot(d_other_signals['select'] * d_other_signals['noise_dist'], 'o', color="k")
            l_axes[1].set_title(f'Other dist {j} - amplifier')

            # Plot details of all selcted bits
            l_axes[2].plot(d_other_signals['select'] + d_origin_signals['select'], color="k")
            l_axes[2].set_title(f'dist {j} of selected bits - amplifier')
            plt.show()
            # plt.savefig(plot_path.format(f'{j}_other_amplifier.png'), format='png')
            # plt.clf()

        if d_criterion['final_criterion'] > selection_thresh:
            sax_I[map_fi.A[:, j], 0] = (d_other_signals['select'] + d_origin_signals['select']) \
                .astype(int)
            level += 1
            print(f"feature {j} selected")

    return sax_I, level


def double_select_amplified_bits(
        sax_inner_left, sax_inner_right, ax_amplified_linputs, ax_amplified_rinputs, ax_base_activations,  map_fi,
        selection_thresh, debug=False,
):

    sax_Il = lil_matrix((len(ax_amplified_linputs), 1), dtype=int)
    sax_Ir = lil_matrix((len(ax_amplified_linputs), 1), dtype=int)
    levell, levelr = 0, 0
    for j in range(map_fi.shape[1]):
        ax_inner_left_sub = sax_inner_left.A[:, map_fi.A[:, j]]
        ax_inner_right_sub = sax_inner_right.A[:, map_fi.A[:, j]]
        ax_origin_mask_left = ~ax_amplified_linputs[map_fi.A[:, j]]
        ax_origin_mask_right = ~ax_amplified_rinputs[map_fi.A[:, j]]

        d_criterion_l, d_origin_signals_l, d_other_signals_l = compute_element_amplifier(
            ax_inner_left_sub, ax_origin_mask_left, ax_base_activations[map_fi.A[:, j]]
        )

        d_criterion_r, d_origin_signals_r, d_other_signals_r = compute_element_amplifier(
            ax_inner_right_sub, ax_origin_mask_right, ax_base_activations[map_fi.A[:, j]]
        )

        print(f'criterion left: {d_criterion_l}')
        print(f'criterion right: {d_criterion_r}')

        if debug:
            fig, l_axes = plt.subplots(1, 3)

            # Plot details of origin bits
            l_axes[0].plot(d_origin_signals_l['bit_dist'], color="k")
            l_axes[0].plot(d_origin_signals_l['noise_dist'], '--', color="k")
            l_axes[0].plot(d_origin_signals_l['select'] * d_origin_signals_l['noise_dist'], 'o', color="k")
            l_axes[0].plot(d_origin_signals_r['bit_dist'], color="b")
            l_axes[0].plot(d_origin_signals_r['noise_dist'], '--', color="b")
            l_axes[0].plot(d_origin_signals_r['select'] * d_origin_signals_r['noise_dist'], 'o', color="b")
            l_axes[0].set_title(f'Origin dist {j} - amplifier')

            # Plot details of other bits
            l_axes[1].plot(d_other_signals_l['bit_dist'], color="k")
            l_axes[1].plot(d_other_signals_l['noise_dist'], '--', color="k")
            l_axes[1].plot(d_other_signals_l['select'] * d_other_signals_l['noise_dist'], 'o', color="k")
            l_axes[1].plot(d_other_signals_r['bit_dist'], color="b")
            l_axes[1].plot(d_other_signals_r['noise_dist'], '--', color="b")
            l_axes[1].plot(d_other_signals_r['select'] * d_other_signals_r['noise_dist'], 'o', color="b")
            l_axes[1].set_title(f'Other dist {j} - amplifier')

            # Plot details of all selcted bits
            l_axes[2].plot(d_other_signals_l['select'] + d_origin_signals_l['select'], color="k")
            l_axes[2].plot(d_other_signals_r['select'] + d_origin_signals_r['select'], color="b")
            l_axes[2].set_title(f'dist {j} of selected bits - amplifier')
            plt.show()

        if d_criterion_l['final_criterion'] > selection_thresh:
            sax_Il[map_fi.A[:, j], 0] = (d_other_signals_l['select'] + d_origin_signals_l['select']) \
                .astype(int)
            levell += 1
            print(f"feature {j} selected for left")

        if d_criterion_r['final_criterion'] > selection_thresh:
            sax_Ir[map_fi.A[:, j], 0] = (d_other_signals_r['select'] + d_origin_signals_r['select']) \
                .astype(int)
            levelr += 1
            print(f"feature {j} selected for right")

    return (sax_Il, levell), (sax_Ir, levelr)


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


def split_drained_graph_new(sax_weight, sax_count, ax_p, ax_r, ax_w, map_fi, target_precision, debug=False, save=False):
    # Get input weights and count
    sax_mask_left, sax_mask_right = (sax_count > 0), (sax_count > 0)
    sax_mask = (sax_weight > 0).multiply(sax_count > 0)

    sax_nom = sax_weight.multiply(sax_mask) - sax_mask.dot(diags(ax_w, format='csc'))
    sax_denom = sax_mask.multiply(sax_count.dot(diags(ax_p + ax_r, format='csc')))
    sax_precision = sax_nom.multiply(sax_denom.astype(float).power(-1))
    sax_precision += (sax_precision != 0).dot(diags(ax_p / (ax_p + ax_r), format='csc'))

    # fuck you
    sax_left = (sax_precision > target_precision).multiply(sax_count > 0)
    sax_right = (sax_precision < target_precision).multiply(sax_count > 0)

    for i in range(map_fi.shape[1]):
        if ((sax_count > 0).astype(int) - (sax_weight > 0))[map_fi[:, i].A[:, 0]].nnz > 0:
            sax_mask_left[map_fi[:, i].A[:, 0]] = sax_left[map_fi[:, i].A[:, 0]]
            sax_mask_right[map_fi[:, i].A[:, 0]] = sax_right[map_fi[:, i].A[:, 0]]

    if debug:
        # Compute precision
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
    return sax_mask_left, sax_mask_right

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