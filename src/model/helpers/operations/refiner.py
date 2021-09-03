# Global import
import numpy as np
from numpy.random import choice
from scipy.sparse import diags, csc_matrix

# Local import
from src.model.helpers.patterns import YalaBasePatterns
from src.model.helpers.data_models import FgComponents
from src.model.helpers.operations.drainer import YalaDrainer


class Refiner(YalaDrainer):
    def __init__(
            self, server, sax_bf_map, drainer_params, min_firing=100, n_update=10, f_candidates=None, perf_plotter=None,
            plot_perf_enabled=False
    ):
        # call parent constructor
        super().__init__(server, sax_bf_map, drainer_params, min_firing, perf_plotter, plot_perf_enabled)

        # Get update params
        self.n_update = n_update
        self.update_candidate_mask(f_candidates)

    def update_candidate_mask(self, f_candidates=None):
        if f_candidates is not None:
            self.mask_candidate = csc_matrix([[
                True if i in f_candidates else False for i in range(self.bf_map.shape[1])
            ]], dtype=bool)

        else:
            self.mask_candidate = csc_matrix(np.ones((1, self.bf_map.shape[1]), dtype=bool))

    def select(self):

        # Compute new inputs, levels and partitions
        sax_inputs = self.select_features(self.fg_mask.I, self.firing_graph.Iw, self.firing_graph.backward_firing['i'])
        ax_levels = sax_inputs.T.dot(self.bf_map).A.sum(axis=1)
        l_partitions = [{**p, "precision": None, "n_firing": None} for p in self.fg_mask.partitions]

        # Create component
        fg_comp = FgComponents(inputs=sax_inputs, partitions=l_partitions, levels=ax_levels)

        if self.plot_perf_enabled:
            self.visualize_fg(YalaBasePatterns.from_fg_comp(fg_comp))

        return fg_comp

    def select_features(self, sax_mask_inputs, sax_drained_weights, sax_count_activations):

        # Get bits with limit_precision
        sax_drained_inputs = self.select_inputs(sax_drained_weights, sax_count_activations)
        sax_no_signal = csc_matrix(np.ones(sax_count_activations.shape, dtype=bool)) - (sax_count_activations > 0)

        # Get new candidate features and their bits cardinality
        ax_mask_features = ~sax_mask_inputs.T.dot(self.bf_map).A
        ax_card_features = self.bf_map.sum(axis=0).A[[0] * ax_mask_features.shape[0], :] * ax_mask_features

        # Get cardinality of each selected feature
        ax_card_selected = (sax_drained_inputs + sax_no_signal).astype(int).T.dot(self.bf_map).A * ax_mask_features

        # Choose new candidate features (cardinality above 0 and lower than feature cardinality)
        ax_mask_selected = self.sample_from_mask((ax_card_selected < ax_card_features) * (0 < ax_card_selected))
        sax_mask_new = self.bf_map.dot(csc_matrix(ax_mask_selected.T))

        # Compute new inputs from refined existing bits and new candidates bits
        sax_inputs = sax_drained_inputs.multiply(sax_mask_inputs) + sax_drained_inputs.multiply(sax_mask_new)

        return sax_inputs

    def sample_from_mask(self, ax_mask):
        # Prepare choice
        (ny, nx), ax_linear = ax_mask.shape, np.arange(ax_mask.shape[1])
        ind_choice = lambda x: list(choice(ax_linear[x], self.n_update, replace=False))

        # Random choice among new candidate features
        l_y_ind = sum([[i] * self.n_update if ax_mask[i, :].sum() > 0 else [] for i in range(ny)], [])
        l_x_ind = sum([ind_choice(ax_mask[i, :]) if ax_mask[i, :].sum() > 0 else [] for i in range(ny)], [])

        # Create new mask features
        ax_mask_sampled = np.zeros((ny, nx), dtype=bool)
        if l_y_ind:
            ax_mask_sampled[l_y_ind, l_x_ind] = True

        return ax_mask_sampled

    def build_patterns(self, component):

        # Create top pattern from comp
        self.fg_mask = YalaBasePatterns.from_fg_comp(component)

        # Create base bottom comp
        self.firing_graph = YalaBasePatterns.from_fg_comp(component)

        # Augment with unexplored candidate features
        sax_candidates = self.bf_map.dot(self.mask_candidate[[0] * len(component), :].T) - \
            self.bf_map.dot(self.firing_graph.I.T.dot(self.bf_map).T)
        self.firing_graph.matrices['Iw'] += sax_candidates.astype(self.firing_graph.matrices['Iw'].dtype)

        # Update levels
        self.firing_graph.levels = np.ones(len(component))

        # Update input matrice's weights
        sax_weights = diags(self.drainer_params.weights, format='csc', dtype=self.firing_graph.matrices['Iw'].dtype)
        self.firing_graph.matrices['Iw'] = self.firing_graph.matrices['Iw'].dot(sax_weights)

        # Update mask draining
        self.firing_graph.matrices['Im'] = self.firing_graph.I

        # Update firing graph from parent
        self.reset_all()
