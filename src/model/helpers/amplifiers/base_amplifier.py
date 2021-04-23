# Global imports
import numpy as np
from scipy.sparse import lil_matrix
from scipy.stats import norm
from matplotlib import pyplot as plt
from functools import lru_cache

# Local import
from src.model.patterns import YalaBasePatterns
from src.model.data_models import AmplificationComponents, FgComponents


class Amplifier():
    def __init__(
            self, server, fi_map, amplification_size, min_size=80, init_level=5, cover_thresh=0.5, size_tresh=0.5,
            max_precision=0.99, ci_value=0.8, gap_fill_length=0, min_level_increment=False, debug=None
    ):
        self.ci_value = ci_value
        self.min_level_increment = min_level_increment
        self.gap_fill_length = gap_fill_length
        self.fi_map = fi_map
        self.server = server
        self.init_level = init_level
        self.amplification_size = amplification_size
        self.min_size = min_size
        self.max_prec = max_precision
        self.cover_thresh = cover_thresh
        self.size_tresh = size_tresh
        self.noise_dist = self.compute_noise_dist()
        self.debug = debug

    @staticmethod
    @lru_cache(maxsize=1024)
    def get_mask_i(n):
        return np.eye(n, )[[2 * i for i in range(int(n / 2))], :]

    @staticmethod
    @lru_cache(maxsize=1024)
    def get_mask_ii(n):
        return np.eye(n, k=1)[[2 * i for i in range(int(n / 2))], :]

    @staticmethod
    def enrich_graph(fg):

        # Build element of enriched graph
        sax_inputs = fg.matrices['Iw'][:, sum([[i, i] for i in range(fg.I.shape[1])], [])]
        levels = np.array(sum([[l, l - 1] for l in fg.levels], []))
        l_partitions = sum([[p, p] for p in fg.partitions], [])

        # Build graph and return it
        return YalaBasePatterns.from_fg_comp(FgComponents(sax_inputs, l_partitions, levels))

    @staticmethod
    def debug_amplify(ax_bit_dist, ax_noise_dist, ax_select, ax_criterion, idx):
        if len(idx) == 1 or ax_bit_dist.shape[0] == 1:
            # Plot details of amplification
            plt.plot(ax_bit_dist[idx[0], :], color="k")
            plt.plot(ax_noise_dist[idx[0], :], '--', color="k")
            plt.plot(ax_select[idx[0], :] * ax_noise_dist[idx[0], :], 'o', color="k")
            plt.title(f'Bit amplification, criterion is {ax_criterion[idx[0]]}')
            plt.show()

        else:
            fig, l_axes = plt.subplots(1, 2)

            # Plot details of amplification
            l_axes[0].plot(ax_bit_dist[idx[0], :], color="k")
            l_axes[0].plot(ax_noise_dist[idx[0], :], '--', color="k")
            l_axes[0].plot(ax_select[idx[0], :] * ax_noise_dist[idx[0], :], 'o', color="k")
            l_axes[0].plot(ax_bit_dist[idx[1], :], color="b")
            l_axes[0].plot(ax_noise_dist[idx[1], :], '--', color="b")
            l_axes[0].plot(ax_select[idx[1], :] * ax_noise_dist[idx[1], :], 'o', color="b")
            l_axes[0].set_title(f'criterion are {ax_criterion[idx]}')

            # Plot details of all selected bits
            l_axes[1].plot(ax_select[idx[0], :], color="k")
            l_axes[1].plot(ax_select[idx[1], :], color="b")
            l_axes[1].set_title('selection')

            plt.show()

    @staticmethod
    def get_binomial_upper_ci(ax_dist, conf, ax_n):
        alpha = 1 - conf
        return ax_dist + (norm.ppf(1 - (alpha / 2)) * np.sqrt(ax_dist * (1 - ax_dist) / ax_n))

    @staticmethod
    def selection_gap_fill(ax_select, n):
        (h, w) = ax_select.shape
        for i in range(w - n):
            # Set ind
            ind = n + i

            # Check for fill value forward backward or if border
            ax_forward_check = ax_select[:, ind] & ax_select[:, ind + 1: ind + n + 2].any(axis=1)
            ax_backward_check = ax_select[:, ind] & ax_select[:, ind - n: ind].any(axis=1)
            ax_border_check = ax_select[:, ind] & np.array([ind == (w - 1)] * h)
            ax_select[:, ind] = ax_forward_check | ax_backward_check | ax_border_check

            ax_select[:, ind] = ax_select[:, ind] | ax_select[:, ind - 1] & ax_select[:, ind: ind + n + 1].any(axis=1)

        return ax_select

    def compute_noise_dist(self):
        sax_x = self.server.next_masked_forward(n=self.amplification_size, update_step=False)
        ax_noise_dist = sax_x.sum(axis=0).A[0] / self.fi_map.dot(sax_x.dot(self.fi_map).sum(axis=0).T).A[:, 0] + 1e-6
        return ax_noise_dist

    def amplify(self, firing_graph):

        # Propagate signal
        amp_comp = self.compute_signals(firing_graph)

        # Select bits
        partial_comp = self.select_bits(amp_comp)

        # Build and return amplified firing_graph
        return partial_comp

    def compute_signals(self, fg):

        enrich_fg = self.enrich_graph(fg)

        # Compute masked activations
        sax_x = self.server.next_masked_forward(n=self.amplification_size, update_step=False)

        # compute inner product between enrich fg activation and bit activation
        sax_inner = enrich_fg.propagate(sax_x).astype(int).T.dot(sax_x)

        # Compute activation count
        ax_norm = sax_inner.dot(self.fi_map).A
        ax_norm = self.get_mask_i(ax_norm.shape[0]).dot(ax_norm).max(axis=1)

        return AmplificationComponents(fg.I, fg.levels, fg.partitions, sax_inner, ax_norm)

    def select_bits(self, amp_comp):

        if len(amp_comp) == 0:
            return None

        ax_levels, sax_I = np.zeros(amp_comp.inputs.shape[1]), lil_matrix(amp_comp.inputs.shape)
        for i in range(self.fi_map.shape[1]):
            # Precompute element to
            ax_inner_sub = amp_comp.bit_inner.A[:, self.fi_map.A[:, i]]
            ax_unselect_mask = ~amp_comp.inputs.A[self.fi_map.A[:, i], :]

            # Qualify amplified bits
            ax_criterion_size, ax_criterion_cover, ax_amplified = self.qualify_bits(
                ax_inner_sub, ax_unselect_mask, self.noise_dist[self.fi_map.A[:, i]], amp_comp.vertex_norm
            )

            # update matrix
            ax_mask = (ax_criterion_cover >= self.cover_thresh) * (ax_criterion_size >= self.size_tresh)
            sax_I[self.fi_map.A[:, i], :] = ax_amplified.T.dot(np.diag(ax_mask))
            ax_levels += (np.ones(ax_criterion_cover.shape) * ax_unselect_mask.any(axis=0) * ax_mask) * 1/2

        if self.min_level_increment:
            ax_levels = ax_levels.clip(amp_comp.levels + 1)

        return FgComponents(inputs=sax_I, partitions=amp_comp.partitions, levels=ax_levels.astype(int))

    def qualify_bits(
            self, ax_inner, ax_origin_mask, ax_noise_dist, vertex_norm
    ):

        # Get dim of problem
        n = ax_inner.shape[0]

        # Get noisy distribution
        ax_norm = self.get_mask_ii(n).dot(ax_inner).sum(axis=1, keepdims=True)
        ax_noise_dist = ax_noise_dist[np.newaxis, :].repeat(ax_norm.shape[0], axis=0)
        ax_noise_dist = self.get_binomial_upper_ci(ax_noise_dist, self.ci_value, ax_norm + 1)

        # Get observation distribution
        ax_obs = (self.get_mask_i(n) + self.get_mask_ii(n)).dot(ax_inner)
        ax_obs -= self.get_mask_i(n).dot(ax_inner) * ax_origin_mask.T
        ax_obs -= self.get_mask_ii(n).dot(ax_inner) * ~ax_origin_mask.T
        ax_obs_norm = ax_obs / (ax_obs.sum(axis=1, keepdims=True) + 1e-6)

        # Get selected_bits
        ax_select = ax_obs_norm > ax_noise_dist

        # Fill gap if necessary
        if self.gap_fill_length > 0 and ax_noise_dist.shape[1] > 10:
            ax_select = self.selection_gap_fill(ax_select, self.gap_fill_length)

        # Compute criterion
        ax_inner_select = self.get_mask_i(n).dot(ax_inner) * ax_select
        ax_criterion_cover = ax_inner_select.sum(axis=1) / (self.get_mask_i(n).dot(ax_inner).sum(axis=1) + 1e-6)
        ax_criterion_size = ax_inner_select.sum(axis=1) / vertex_norm

        if self.debug is not None:
            self.debug_amplify(
                ax_obs_norm, ax_noise_dist, ax_select, ax_criterion_cover, self.debug['indices']
            )

        return ax_criterion_size.round(1), ax_criterion_cover.round(1), ax_select
