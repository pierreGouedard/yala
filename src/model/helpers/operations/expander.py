# Global import
import numpy as np
from scipy.sparse import diags, csc_matrix
from numpy.random import choice
from matplotlib import pyplot as plt

# Local import
from src.model.helpers.patterns import YalaBasePatterns
from src.model.helpers.data_models import FgComponents
from src.model.helpers.operations.drainer import YalaDrainer


class Expander(YalaDrainer):
    """Expander"""

    def __init__(
            self, server, sax_bf_map, drainer_params, min_firing=100, n_update=1, perf_plotter=None, plot_perf_enabled=False,
            plot_ind=0
    ):
        # call parent constructor
        super().__init__(server, sax_bf_map, drainer_params, min_firing, perf_plotter, plot_perf_enabled)

    def __visualize_multi_expansion(self, sax_i_new, ax_selection_mask):
        # for each selected bounds
        partitions = [{'label': 0} for _ in range(sax_i_new.shape[1])]
        ax_feature_mask = self.fg_mask.I.T.dot(self.bf_map).A
        for i in range(ax_feature_mask.shape[0]):
            for ind in ax_feature_mask[i, :].nonzero()[0]:
                d_info = {"vertex_id": i, "feature_ind": ind, "is_selected": ax_selection_mask[i, ind]}

                ax_mask = ax_feature_mask[i, :].copy() ^ np.eye(ax_feature_mask.shape[1], dtype=bool)[:, ind]
                fg_free_space = YalaBasePatterns.from_fg_comp(FgComponents(
                    inputs=self.bf_map.dot(csc_matrix(ax_mask).T).multiply(self.fg_mask.I[:, i]).astype(int),
                    partitions=partitions, levels=np.array([ax_feature_mask.sum() - 1])
                ))
                fg_old_bounds = YalaBasePatterns.from_fg_comp(FgComponents(
                    inputs=self.bf_map[:, ind].multiply(self.fg_mask.I[:, i]).astype(int), partitions=partitions,
                    levels=np.array([1])
                ))
                fg_new_bounds = YalaBasePatterns.from_fg_comp(FgComponents(
                    inputs=self.bf_map[:, ind].multiply(sax_i_new[:, i].astype(int)).astype(int), partitions=partitions,
                    levels=np.array([1])
                ))

                self.__visualize_expansion(fg_free_space, fg_old_bounds, fg_new_bounds, d_info)

    def __visualize_expansion(self, fg_free_space, fg_old_bounds, fg_new_bounds, d_info):
        # Get signals
        sax_x = self.server.get_sub_forward(self.perf_plotter.indices)
        ax_free_space = fg_free_space.propagate(sax_x).A[:, 0]
        ax_old_bounds, ax_new_bounds = fg_old_bounds.propagate(sax_x).A[:, 0], fg_new_bounds.propagate(sax_x).A[:, 0]

        # Plot perf viz
        fig, (ax_old, ax_new) = plt.subplots(1, 2)
        plt.suptitle(f"""
            Figure of expansion of vertex {d_info['vertex_id']} feature {d_info['feature_ind']}: 
            selected = {d_info['is_selected']}
        """)

        ax_old.scatter(
            self.perf_plotter.x[self.perf_plotter.y > 0, 0], self.perf_plotter.x[self.perf_plotter.y > 0, 1],
            c='g',  alpha=0.05
        )
        ax_old.scatter(
            self.perf_plotter.x[ax_free_space > 0, 0], self.perf_plotter.x[ax_free_space > 0, 1], c='r', alpha=0.1
        )
        ax_old.scatter(
            self.perf_plotter.x[ax_old_bounds > 0, 0], self.perf_plotter.x[ax_old_bounds > 0, 1],
            c='b', alpha=0.1
        )
        ax_new.scatter(
            self.perf_plotter.x[self.perf_plotter.y > 0, 0], self.perf_plotter.x[self.perf_plotter.y > 0, 1],
            c='g',  alpha=0.05
        )
        ax_new.scatter(
            self.perf_plotter.x[ax_free_space > 0, 0], self.perf_plotter.x[ax_free_space > 0, 1], c='r', alpha=0.1
        )
        ax_new.scatter(
            self.perf_plotter.x[ax_new_bounds > 0, 0], self.perf_plotter.x[ax_new_bounds > 0, 1], c='b', alpha=0.1
        )

        plt.show()

    def select(self):

        # Compute new inputs, levels and partitions
        sax_inputs, ax_n_firing = self.merge_inputs(
            self.fg_mask.I, self.firing_graph.Iw, self.firing_graph.backward_firing['i']
        )
        ax_levels = sax_inputs.T.dot(self.bf_map).A.sum(axis=1)
        l_partitions = [
            {**p, "precision": self.drainer_params.precisions[i], "n_firing": ax_n_firing[i]}
            for i, p in enumerate(self.fg_mask.partitions)
        ]

        # Create component
        fg_comp = FgComponents(inputs=sax_inputs, partitions=l_partitions, levels=ax_levels)

        if self.plot_perf_enabled:
            self.visualize_fg(YalaBasePatterns.from_fg_comp(fg_comp))

        return fg_comp

    def merge_inputs(self, sax_mask_inputs, sax_drained_weights, sax_count_activations):
        # Get selected bits and and merge with base bit
        sax_drained_inputs = self.select_inputs(sax_drained_weights, sax_count_activations)
        sax_merged_inputs = sax_mask_inputs + sax_drained_inputs
        sax_no_signal = csc_matrix(np.ones(sax_count_activations.shape, dtype=bool)) - (sax_count_activations > 0)

        # Compute nb firing
        ax_n_firing = sax_count_activations.sum(axis=0).A[0].astype(int)

        # Get new candidate features and their bits cardinality
        ax_mask_features = sax_mask_inputs.T.dot(self.bf_map).A
        ax_card_features = self.bf_map.sum(axis=0).A[[0] * ax_mask_features.shape[0], :] * ax_mask_features

        # Select bounds to remove
        ax_card_no_signal = (sax_mask_inputs + sax_no_signal).astype(int).T.dot(self.bf_map).A
        ax_mask_selected = ax_card_no_signal < ax_card_features

        # Visualize result of expansion
        if self.plot_perf_enabled:
            self.__visualize_multi_expansion(sax_merged_inputs, ax_mask_selected)

        # Compute new inputs from mask of feature
        sax_inputs = sax_merged_inputs.multiply(self.bf_map.dot(csc_matrix(ax_mask_selected.T)))

        return sax_inputs, ax_n_firing

    def build_patterns(self, component):
        # Create mask pattern from comp
        mask_component = component.copy()
        mask_component.levels = mask_component.levels - 1
        self.fg_mask = YalaBasePatterns.from_fg_comp(mask_component)

        # Create base bottom comp
        sax_mask = self.bf_map.dot(self.bf_map.T.dot(component.inputs))
        base_component = component.complement(sax_mask)
        base_component.levels = np.ones(len(base_component))
        self.firing_graph = YalaBasePatterns.from_fg_comp(base_component)

        # Update base matrice input's weights
        sax_weights = diags(self.drainer_params.weights, format='csc', dtype=self.firing_graph.matrices['Iw'].dtype)
        self.firing_graph.matrices['Iw'] = self.firing_graph.matrices['Iw'].dot(sax_weights)

        # Update mask draining
        self.firing_graph.matrices['Im'] = self.firing_graph.I

        # Update firing graph from parent
        self.reset_all()
