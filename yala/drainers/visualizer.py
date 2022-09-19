# Global import
from matplotlib import pyplot as plt

# Local import
from yala.firing_graph import YalaFiringGraph
from .drainer import YalaDrainer


class Visualizer(YalaDrainer):
    """Visual"""
    def __init__(
            self, server, bitmap, drainer_params, n_bounds=2, perf_plotter=None,
            plot_perf_enabled=False, advanced_plot_perf_enabled=False
    ):
        # Get visualisation parameters
        self.perf_plotter = perf_plotter
        self.plot_perf_enabled = plot_perf_enabled
        self.advanced_plot_perf_enabled = advanced_plot_perf_enabled

        # Call parent constructor
        super().__init__(server, bitmap, drainer_params, n_bounds)

    def visualize_shaping(self, drain_components, mask_components, base_components):
        sax_x = self.server.get_sub_forward(self.perf_plotter.indices)
        ax_drain_y = YalaFiringGraph.from_fg_comp(drain_components).propagate(sax_x).A
        ax_mask_y = YalaFiringGraph.from_fg_comp(mask_components).propagate(sax_x).A
        ax_base_y = YalaFiringGraph.from_fg_comp(base_components).propagate(sax_x).A

        for i in range(ax_drain_y.shape[1]):
            # GOT
            plt.scatter(
                self.perf_plotter.x[self.perf_plotter.y > 0, 0], self.perf_plotter.x[self.perf_plotter.y > 0, 1],
                c='g', alpha=0.3, marker='s', label='got (True)'
            )

            # Mask space
            plt.scatter(
                self.perf_plotter.x[ax_mask_y[:, i] > 0, 0], self.perf_plotter.x[ax_mask_y[:, i] > 0, 1],
                c='b', alpha=0.5, marker="+", label='Mask space', s=50
            )

            # Search space
            plt.scatter(
                self.perf_plotter.x[(ax_drain_y * ax_mask_y)[:, i] > 0, 0],
                self.perf_plotter.x[(ax_drain_y * ax_mask_y)[:, i] > 0, 1], c='r',
                alpha=0.5, marker='+', label='Search space', s=50
            )

            # Post shaping space
            plt.scatter(
                self.perf_plotter.x[(ax_drain_y * ax_mask_y * ax_base_y)[:, i] > 0, 0],
                self.perf_plotter.x[(ax_drain_y * ax_mask_y * ax_base_y)[:, i] > 0, 1],
                c='k', alpha=0.2, marker="s", label='Post shaping space'
            )

            plt.title(f'Component {i}: Shaping visualisation')
            plt.legend()
            plt.show()

    def visualize_comp(self, components):
        if self.plot_perf_enabled is None:
            raise ValueError('Impossible to visualize firing graph: not plot perf.')

        for i in range(len(components)):
            # Get masked activations
            sax_x = self.server.get_sub_forward(self.perf_plotter.indices)

            ax_yhat = YalaFiringGraph.from_fg_comp(components).propagate(sax_x).A[:, [i]]

            # Plot perf viz
            self.perf_plotter(ax_yhat)



