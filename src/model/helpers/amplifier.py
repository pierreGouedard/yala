# Global imports
import numpy as np
from scipy.sparse import diags, lil_matrix
from scipy.stats import norm
from matplotlib import pyplot as plt
from functools import lru_cache

# Local import
from src.model.patterns import YalaBasePatterns


class Amplifier():
    def __init__(
            self, server, fi_map, amplification_size, min_size=100, init_level=5, select_thresh=0.5, gap_fill_length=2,
            debug=None
    ):
        self.fi_map = fi_map
        self.server = server
        self.init_level = init_level
        self.amplification_size = amplification_size
        self.min_size = min_size
        self.select_thresh = select_thresh
        self.gap_fill_length = gap_fill_length
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
    def enrich_partition(i, p):
        return [{**p, "indices": 2 * i, "output_id": 2 * i}, {**p, "indices": 2 * i + 1, "output_id": 2 * i + 1}]

    @staticmethod
    def debug_amplify(total_norm, ax_noise_dist, ax_select, ax_criterion, ind, k):
        plt.plot(total_norm[ind, :], color="k")
        plt.plot(ax_noise_dist[ind, :], '--', color="k")
        plt.plot(ax_select[ind, :] * ax_noise_dist[ind, :], 'o', color = "k")
        plt.title(f'Bit amplification for feature {k}, indice {ind}, criterion is {ax_criterion[ind]}')
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
        self.server.next_forward(n=self.amplification_size)
        sax_i = self.server.sax_data_forward[(1 - self.server.sax_mask_forward.A[:, 0] > 0), :]
        ax_noise_dist = sax_i.sum(axis=0).A[0] / self.fi_map.dot(sax_i.dot(self.fi_map).sum(axis=0).T).A[:, 0]
        return ax_noise_dist

    def enrich_graph(self, fg):

        # Build element of enriched graph
        sax_inputs = fg.matrices['Iw'][:, sum([[i, i] for i in range(fg.I.shape[1])], [])]
        levels = np.array(sum([[l, l - 1] for l in fg.levels], []))
        l_partitions = sum([self.enrich_partition(i, p) for i, p in enumerate(fg.partitions)], [])

        # Build graph and return it
        return YalaBasePatterns.from_input_matrix(sax_inputs, l_partitions, levels)

    def sample_and_amplify(self, n_sample=2):
        # Get randomly sampled input through servers and build partitions
        sax_inputs = self.server.get_random_samples(n_sample).T
        l_partitions = [{'indices': i, 'output_id': i, 'label': 0, 'precision': 0} for i in range(n_sample)]

        # Build sampled graph and amplify it
        fg = YalaBasePatterns.from_input_matrix(sax_inputs, l_partitions, np.array([self.init_level] * n_sample))
        return self.amplify(fg, parent_comp=None)

    def amplify(self, firing_graph, parent_comp=None):

        # Propagate signal
        sax_inner, ax_count = self.compute_signals(firing_graph)

        # Select bits
        sax_I, ax_levels = self.select_bits(firing_graph, sax_inner, ax_count)

        # Build and return amplified firing_graph
        return YalaBasePatterns.from_input_matrix(sax_I, {}, ax_levels.astype(int))

    def compute_signals(self, fg):

        enrich_fg = self.enrich_graph(fg)

        # Compute masked activations
        self.server.next_forward(n=self.amplification_size)
        sax_x, sax_mask = self.server.sax_data_forward, self.server.sax_mask_forward

        if sax_mask.nnz > 0:
            sax_x = diags(~(sax_mask.A[:, 0] > 0), dtype=bool).dot(sax_x)

        # compute inner product between enrich fg activation and bit activation
        sax_inner = enrich_fg.propagate(sax_x).astype(int).T.dot(sax_x)

        # Compute activation count
        ax_norm = sax_inner.dot(self.fi_map).A
        ax_norm = self.get_mask_i(ax_norm.shape[0]).dot(ax_norm).max(axis=1)

        return sax_inner, ax_norm

    def select_bits(self, fg, sax_inner, ax_norm):

        # TODO: for each candidate, if one of the left / right part has sufficient activations, just removed the other
        #   part and go on with the survivor, If both ahas not sufficient activation, merge them to get back parent
        #    and return it as "complete" to refine !
        # l_pidx = sum([[2 * i, 2 * i + 1] for i in np.arange(ax_count.shape[0])[ax_count > self.min_size]], [])
        # sax_inner_partials = sax_inner[l_pidx]

        ax_levels, sax_I = np.zeros(fg.I.shape[1]), lil_matrix(fg.I.shape)
        for i in range(self.fi_map.shape[1]):
            # Precompute element to
            ax_inner_sub = sax_inner.A[:, self.fi_map.A[:, i]]
            ax_unselect_mask = ~fg.I.A[self.fi_map.A[:, i], :]

            # Add feature info to debugger
            if self.debug is not None:
                d_debug = {**self.debug, 'feature': i}
            else:
                d_debug = None

            # Qualify amplified bits
            ax_size, ax_criterions, ax_amplified = self.qualify_bits(
                ax_inner_sub, ax_unselect_mask, self.noise_dist[self.fi_map.A[:, i]], d_debug=d_debug
            )

            # update matrix
            ax_mask = (ax_criterions > self.select_thresh) * (ax_size / ax_norm > self.select_thresh)
            sax_I[self.fi_map.A[:, i], :] = ax_amplified.T.dot(np.diag(ax_mask))
            # TODO: check ether this "new" criterion worth it may be a criterion of all /2 may be better.
            ax_levels += (ax_criterions * ax_unselect_mask.any(axis=0) * ax_mask)

        return sax_I, ax_levels

    def qualify_bits(
            self, ax_inner, ax_origin_mask, ax_noise_dist, ci_select=0.8, gap_fill=False, d_debug=None
    ):

        # Get dim of problem
        n = ax_inner.shape[0]

        # Get noisy distribution
        ax_norm = self.get_mask_i(n).dot(ax_inner).sum(axis=1, keepdims=True)
        ax_noise_dist = ax_noise_dist[np.newaxis, :].repeat(ax_norm.shape[0], axis=0)
        ax_noise_dist = self.get_binomial_upper_ci(ax_noise_dist, ci_select, ax_norm)

        # Get observation distribution
        ax_obs = (self.get_mask_i(n) + self.get_mask_ii(n)).dot(ax_inner)
        ax_obs -= self.get_mask_i(n).dot(ax_inner) * ax_origin_mask.T
        ax_obs += self.get_mask_ii(n).dot(ax_inner) * ~ax_origin_mask.T
        ax_obs_norm = ax_obs / ax_obs.sum(axis=1, keepdims=True)

        # Get selected_bits
        ax_select = ax_obs_norm > ax_noise_dist

        # Fill gap if necessary
        if gap_fill and len(ax_noise_dist) < 10:
            ax_select = self.selection_gap_fill(ax_select, self.gap_fill_length, len(ax_noise_dist))

        # Compute criterion
        ax_inner_select = self.get_mask_i(n).dot(ax_inner) * ax_select
        ax_criterion = ax_inner_select.sum(axis=1) / self.get_mask_i(n).dot(ax_inner).sum(axis=1)
        ax_size = ax_inner_select.sum(axis=1)

        if d_debug is not None:
            self.debug_amplify(
                ax_obs_norm, ax_noise_dist, ax_select, ax_criterion, d_debug['indice'], d_debug['feature']
            )

        return ax_size, ax_criterion, ax_select
