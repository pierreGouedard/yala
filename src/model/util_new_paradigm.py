# Global import
from scipy.stats import norm
from scipy.sparse import lil_matrix, diags
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import asdict

# Local import
from src.model.patterns import YalaBasePatterns
from .utils import set_feedbacks
from .data_models import DrainerFeedbacks, DrainerParameters
from firing_graph.solver.drainer import FiringGraphDrainer




def compute_element_amplifier(ax_inner, ax_origin_mask, ax_base_activations, ci_select=0.8, gap_fill=False, gap_fill_len=2):

    n_vertex = ax_inner[0, ax_origin_mask].sum() + ax_inner[0, ~ax_origin_mask].sum()

    # Get noisy distribution
    ax_noise_dist = ax_base_activations / ax_base_activations.sum()
    ax_noise_dist = get_binomial_upper_ci(ax_noise_dist, ci_select, ax_inner[1, :].sum())

    # Get observation distribution
    ax_obs = ax_inner.sum(axis=0) - ((ax_inner[0, :] * ax_origin_mask) + (ax_inner[1, :] * ~ax_origin_mask))
    ax_obs_dist = ax_obs / ax_obs.sum()
    ax_select = ax_obs_dist > ax_noise_dist

    # Gap fill if necessary
    if gap_fill and len(ax_base_activations) > 10:
        for i in range(len(ax_select) - gap_fill_len):
            ind = gap_fill_len + i
            check_after = ax_select[ind] and ax_select[ind + 1: ind + gap_fill_len + 2].any()
            check_before = ax_select[ind] and ax_select[ind - gap_fill_len: ind].any()
            right_border = ax_select[ind] and (ind == len(ax_select) - 1)

            ax_select[ind] = check_before | check_after | right_border
            ax_select[ind] |= ax_select[ind - 1] and ax_select[ind: ind + gap_fill_len + 1].any()

    # Compute criterion and save signals
    d_criterions = dict(
        n_selected=ax_inner[0, ax_select].sum(), n_vertex=n_vertex,
        criterion=round(ax_inner[0, ax_select].sum() / n_vertex, 3),
        idx_select=np.arange(ax_origin_mask.shape[0])[~ax_origin_mask],
    )
    d_signals = dict(obs_dist=ax_obs_dist, noise_dist=ax_noise_dist, selection=ax_select)

    return d_criterions, d_signals


def amplify_debug_display(d_signals, n, add=None):

    # Plot details of origin bits
    plt.plot(d_signals['obs_dist'], color="k")
    plt.plot(d_signals['noise_dist'], '--', color="k")
    plt.plot(d_signals['selection'] * d_signals['noise_dist'], 'o', color="k")
    if add is not None:
        plt.plot(add * d_signals['noise_dist'], '+', color="b")
    plt.title(f'Bit amplification for feature {n}')
    plt.show()


def amplify_bits(sax_inner, ax_inputs, ax_base_activations, map_fi, ax_thresh, debug=False):

    # Init amplified vertex params
    sax_I, level = lil_matrix((len(ax_inputs), 1), dtype=int), 0
    n_vertex = sax_inner[0, :].dot(map_fi).max()
    for i in range(map_fi.shape[1]):
        ax_inner_sub, ax_origin_mask = sax_inner.A[:, map_fi.A[:, i]], ~ax_inputs[map_fi.A[:, i]]

        from_parent = (~ax_origin_mask).any()
        d_criterions, d_signals = compute_element_amplifier(
            ax_inner_sub, ax_origin_mask, ax_base_activations[map_fi.A[:, i]]
        )

        # Display info and debug
        if debug:
            print(f'criterion: {d_criterions}')
            amplify_debug_display(d_signals, i)

        # build new vertex's inputs
        if d_criterions['criterion'] > ax_thresh[i] and d_criterions['n_selected'] / n_vertex > 0.5:
            sax_I[map_fi.A[:, i], 0] = d_signals['selection'].astype(int)

            if from_parent:
                level += d_criterions['criterion']

    return sax_I, int(level), ax_thresh


def final_bit_selection(fg, map_fi, X, ax_base_activations, ax_thresh, margin_lvl=2, debug=False):
    stop = False
    while not stop:
        sax_x = fg.propagate(X).tocsc()
        if sax_x[:, 0].sum() > 10:
            sax_inner = sax_x.astype(int).T.dot(X)
            stop = True
            break
        fg.levels -= 1

    n_vertex = sax_inner[0, :].dot(map_fi).max()
    sax_I = lil_matrix((fg.I.shape[0], 1), dtype=int)

    count_added, count_removed, count_features, level = 0, 0, 0, 0
    for j in range(map_fi.shape[1]):
        from_parent = (fg.I.A[:, 0][map_fi.A[:, j]]).any()

        d_criterion, d_signals = compute_element_amplifier(
            sax_inner.A[:, map_fi.A[:, j]], ~fg.I.A[:, 0][map_fi.A[:, j]], ax_base_activations[map_fi.A[:, j]],
            ci_select=0.5, gap_fill=True
        )

        if debug:
            print(d_criterion)
            amplify_debug_display(d_signals, j, add=fg.I.A[:, 0][map_fi.A[:, j]])

        if d_criterion['criterion'] >= 0.8 and d_criterion['n_selected'] / n_vertex > 0.5:
            sax_I[map_fi.A[:, j], 0] = d_signals['selection'].astype(int)
            level += d_criterion['criterion']
            count_features += 1

            count_added += int((not from_parent))

        else:
            count_removed += int(from_parent)

    stop = (count_added == 0) and (count_removed == 0)

    if stop:
        return sax_I, count_features - margin_lvl, ax_thresh, True
    else:
        return sax_I, int(level), ax_thresh + 1e-1, False


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


def get_drainer_firing_graph(sax_I, level, map_fi, n_drain=100):
    firing_graph = YalaBasePatterns.from_input_matrix(
        sax_I, [{'indices': 0, 'output_id': 0, 'label': 0, 'precision': 0}], np.array([level])
    )

    sax_input_features = sax_I.astype(bool).T.dot(map_fi)
    if n_drain < sax_input_features.nnz:

        # Set draining mask to True only for features selected for draining
        ax_idx = np.random.choice(sax_input_features.nonzero()[1], sax_input_features.nnz - n_drain, replace=False)
        sax_input_features[0, ax_idx] = False
        firing_graph.matrices['Im'] = sax_I.multiply(map_fi.dot(sax_input_features.T))

    else:
        firing_graph.matrices['Im'] = sax_I

    return firing_graph


def split_drained_graph(sax_weight, sax_count):

    # Get input weights and count
    sax_common = ((sax_weight > 0).astype(int) - (sax_count > 0).astype(int) > 0)
    sax_left_mask = (sax_weight > 0).multiply((sax_count > 0))
    sax_right_mask = ((sax_count > 0).astype(int) - sax_left_mask.astype(int) > 0)

    return sax_left_mask + sax_common, sax_right_mask + sax_common


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


def show_significance_plot(fg, map_fi, X):

    n_features = fg.I[:, 0].T.dot(map_fi).sum()
    ax_activations = np.zeros((2, n_features))
    for i in range(n_features - 1):
        # Random firing graph
        test_fg = create_random_fg(fg, map_fi, n_features - i)
        ax_activations[0, i] = test_fg.propagate(X).sum()

        fg.levels[0] = n_features - i
        ax_activations[1, i] = fg.propagate(X).tocsc()[:, 0].sum()

    plt.plot(ax_activations[0, :], color='k')
    plt.plot(ax_activations[1, :], color='b')
    plt.show()


def show_activation_stats(fg, map_fi, X, y):
    n_features = fg.I.T.dot(map_fi).sum()
    sax_x_final = fg.propagate(X).tocsc()
    prec = sax_x_final[:, 0].T.astype(int).dot(y).A / sax_x_final[:, 0].sum()
    print(
        f"Final fg with # feature: {n_features}, level: {fg.levels[0]}, prec: {prec}, "
        f"# activate: {sax_x_final[:, 0].sum()} times"
    )


def show_draining_stats(fg, map_fi, X, y):

    sax_x = fg.propagate(X).tocsc()
    sax_inner = X.multiply(sax_x[:, [0] * X.shape[1]])
    ax_x = sax_inner.sum(axis=0).A[0]
    ax_prec = sax_inner.T.astype(int).dot(y).A[:, 0] / (ax_x + 1e-6)

    for j in range(map_fi.shape[1]):
        ax_prec_sub = ax_prec[map_fi.A[:, j]]

        # Plot prec
        plt.plot(ax_prec_sub, color="k")
        plt.plot(fg.I.A[map_fi.A[:, j], 0], "--", color='b')
        plt.plot(ax_x[map_fi.A[:, j]] / ax_x[map_fi.A[:, j]].sum(), "*", color='k')

        plt.title(f'Precision bits for feature {j}')
        plt.show()


def reselect_graph(fg, map_fi, X):
    ax_inner = fg.propagate(X).astype(int).T.dot(X).A[0]
    sax_I, level = lil_matrix(fg.I.shape), 0

    for i in range(map_fi.shape[1]):
        ax_dist_activation = ax_inner[map_fi.A[:, i]] / ax_inner[map_fi.A[:, i]].sum()

        # Show input and activation
        plt.plot(fg.I.A[map_fi.A[:, i], 0] * ax_dist_activation.mean(), "--", color='b')
        plt.plot(ax_dist_activation, color='k')
        plt.show()

        # Ask if we should tke it ?
        takeit = input("Take the input ?: ")

        if takeit == 'yes':
            sax_I[map_fi.A[:, i], 0] = (ax_dist_activation > 1e-2).astype(int)
            level += 1

    return sax_I, level


def show_diff_fg(fga, fgb, map_fi):
    for j in range(map_fi.shape[1]):

        ax_ina = fga.I.A[:, 0][map_fi.A[:, j]]
        ax_inb = fgb.I.A[:, 0][map_fi.A[:, j]]

        # Plot diff inputs
        plt.plot(ax_ina, color="k")
        plt.plot(ax_inb, '--', color="b")
        plt.title(f'Selected bits for feature {j}')
        plt.show()


