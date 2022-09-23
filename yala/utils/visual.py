# Global import
from matplotlib import pyplot as plt
import numpy as np

# Local import
from yala.firing_graph import YalaFiringGraph
from yala.utils.data_models import FgComponents


class Visualizer:
    """Visual"""
    def __init__(self, perf_plotter, bitmap):
        # Get visualisation parameters
        self.perf_plotter = perf_plotter
        self.bitmap = bitmap

    def visualize_comp(self, comps, server):
        for i in range(len(comps)):
            # Get masked activations
            sax_x = server.get_sub_forward(self.perf_plotter.indices)

            ax_yhat = YalaFiringGraph.from_comp(comps).seq_propagate(sax_x).A[:, [i]]

            # Plot perf viz
            self.perf_plotter(ax_yhat)

    def visualize_shaping(self, sax_drain_inputs, sax_other_inputs, base_components, server):
        # Build firing graphs
        drain_fg, mask_fg, base_fg = self.build_fg(sax_drain_inputs, sax_other_inputs, base_components)

        # Propagate data
        sax_x = server.get_sub_forward(self.perf_plotter.indices)
        ax_drain_y = drain_fg.seq_propagate(sax_x).A
        ax_mask_y = mask_fg.seq_propagate(sax_x).A
        ax_base_y = base_fg.seq_propagate(sax_x).A

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

    def build_fg(self, sax_drain_inputs, sax_other_inputs, base_components):

        drain_components = FgComponents(
            inputs=sax_other_inputs, mask_inputs=sax_other_inputs, levels=np.ones(sax_drain_inputs.shape[1]),
            partitions=base_components.partitions
        )
        other_components = FgComponents(
            inputs=sax_drain_inputs, mask_inputs=sax_drain_inputs, partitions=base_components.partitions,
            levels=self.bitmap.b2f(sax_other_inputs).sum(axis=1).A[:, 0]
        )
        return (
            YalaFiringGraph.from_comp(drain_components),
            YalaFiringGraph.from_comp(other_components),
            YalaFiringGraph.from_comp(base_components)
        )




