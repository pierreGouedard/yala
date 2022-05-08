# Global import
import numpy as np
from matplotlib import pyplot as plt

# Local import
from src.model.core.firing_graph import YalaFiringGraph
from src.model.core.data_models import FgComponents
from src.model.core.drainers.drainer import YalaDrainer


class Visualizer(YalaDrainer):
    """Visual"""
    def __init__(
            self, server, bitmap, drainer_params, min_firing=100, n_bounds=2, perf_plotter=None,
            plot_perf_enabled=False, advanced_plot_perf_enabled=False
    ):
        # Get visualisation parameters
        self.perf_plotter = perf_plotter
        self.plot_perf_enabled = plot_perf_enabled
        self.advanced_plot_perf_enabled = advanced_plot_perf_enabled

        # Call parent constructor
        super().__init__(server, bitmap, drainer_params, min_firing, n_bounds)

    def select(self):
        fg_comp = YalaDrainer.select(self)

        if self.plot_perf_enabled:
            self.visualize_fg(YalaFiringGraph.from_fg_comp(fg_comp))

        return fg_comp

    def visualize_shapes(self, fg, mask_fg):
        # Get masked activations
        sax_x = self.server.get_sub_forward(self.perf_plotter.indices)
        ax_y, ax_mask_y = fg.propagate(sax_x).A, mask_fg.propagate(sax_x).A

        for i in range(ax_y.shape[1]):
            # Plot perf viz
            plt.scatter(
                self.perf_plotter.x[ax_y[:, i] > 0, 0], self.perf_plotter.x[ax_y[:, i] > 0, 1], c='r', alpha=0.25,
                label='to_drain_area'
            )

            plt.scatter(
                self.perf_plotter.x[ax_mask_y[:, i] > 0, 0], self.perf_plotter.x[ax_mask_y[:, i] > 0, 1], c='b',
                alpha=0.25, label='mask_area'
            )

            plt.title(f'Component {i}: mask area vs to drain space - stage: {fg.partitions[i]["stage"]}')
            plt.legend()
            plt.show()

    def visualize_fg(self, firing_graph):
        if self.plot_perf_enabled is None:
            raise ValueError('Impossible to visualize firing graph: not plot perf.')

        # Get masked activations
        sax_x = self.server.get_sub_forward(self.perf_plotter.indices)
        ax_yhat = firing_graph.propagate(sax_x).A

        # Plot perf viz
        self.perf_plotter(ax_yhat)

    def visualize_multi_selection(self, sax_new_inputs, ax_selection_mask):

        # for each vertex / original support features
        mask_comp = FgComponents(inputs=self.fg_mask.I, partitions=self.fg_mask.partitions, levels=self.fg_mask.levels)

        ax_feature_mask = self.bitmap.b2f(self.original_inputs).A
        for i, sub_mask_comp in enumerate(mask_comp):
            fg_search_space = YalaFiringGraph.from_fg_comp(sub_mask_comp)

            for ind in ax_feature_mask[i, :].nonzero()[0]:
                # Compute sub inputs for original / new bounds
                sax_ori_sub_inputs = self.bitmap.bf_map[:, ind].multiply(self.original_inputs[:, i]).astype(int)
                sax_new_sub_inputs = self.bitmap.bf_map[:, ind].multiply(sax_new_inputs[:, i]).astype(int)

                fg_ori_bounds = YalaFiringGraph.from_fg_comp(FgComponents(
                    inputs=sax_ori_sub_inputs, partitions=[{'label': 0}], levels=np.array([1])
                ))
                fg_new_bounds = YalaFiringGraph.from_fg_comp(FgComponents(
                    inputs=sax_new_sub_inputs, partitions=[{'label': 0}], levels=np.array([1])
                ))

                # Visualize
                self.visualize_selection(
                    fg_search_space, fg_ori_bounds, fg_new_bounds,
                    {
                        "v_ind": i, "f_ind": ind, "is_support": ax_selection_mask[i, ind],
                        'stage': type(self).__name__
                    }
                )

    def visualize_selection(self, fg_search_space, fg_ori_bounds, fg_new_bounds, d_info):
        # Get signals
        sax_x = self.server.get_sub_forward(self.perf_plotter.indices)
        ax_search_space = fg_search_space.propagate(sax_x).A[:, 0]
        ax_ori_bounds = fg_ori_bounds.propagate(sax_x).A[:, 0]
        ax_new_bounds = fg_new_bounds.propagate(sax_x).A[:, 0]

        # Plot perf viz
        plt.suptitle(f"""
            Figure of {d_info['stage']} of vertex {d_info['v_ind']} feature {d_info['f_ind']}: 
            support = {d_info['is_support']} 
        """)

        plt.scatter(
            self.perf_plotter.x[self.perf_plotter.y > 0, 0], self.perf_plotter.x[self.perf_plotter.y > 0, 1],
            c='g',  alpha=0.05, label='got (True)'
        )
        plt.scatter(
            self.perf_plotter.x[ax_search_space > 0, 0], self.perf_plotter.x[ax_search_space > 0, 1],
            c='r', alpha=0.1, label='Search space'
        )
        plt.scatter(
            self.perf_plotter.x[ax_ori_bounds > 0, 0], self.perf_plotter.x[ax_ori_bounds > 0, 1],
            c='b', alpha=0.1, label='Pre-draining space'
        )
        plt.scatter(
            self.perf_plotter.x[ax_new_bounds > 0, 0], self.perf_plotter.x[ax_new_bounds > 0, 1],
            c='k', alpha=0.1, label='Selected space'
        )
        plt.legend()
        plt.show()



