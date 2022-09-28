# Global import
from matplotlib import pyplot as plt
import numpy as np

# Local import
from yala.firing_graph import YalaFiringGraph
from yala.linalg.spmat_op import bounds


class Visualizer:
    """Visual"""
    def __init__(self, perf_plotter):
        # Get visualisation parameters
        self.perf_plotter = perf_plotter

    def visualize_comp(self, comps, server):
        for i in range(len(comps)):
            # Get masked activations
            sax_x = server.get_sub_forward(self.perf_plotter.indices)

            ax_yhat = YalaFiringGraph.from_comp(comps).seq_propagate(sax_x).A[:, [i]]

            # Plot perf viz
            self.perf_plotter(ax_yhat)

    def visualize_cleaning(self, server, comps, cleaned_comps):
        sax_x = server.get_sub_forward(self.perf_plotter.indices)

        ax_y = YalaFiringGraph.from_comp(
            comps.update(levels=np.ones(len(comps)), inputs=bounds(comps.inputs, server.bitmap))
        ).seq_propagate(sax_x).A

        ax_clean_y = YalaFiringGraph.from_comp(
            cleaned_comps.update(
                levels=np.ones(len(cleaned_comps)), inputs=bounds(cleaned_comps.inputs, server.bitmap)
            )
        ).seq_propagate(sax_x).A

        for i in range(ax_y.shape[1]):
            # Before cleaning bounds
            plt.scatter(
                self.perf_plotter.x[ax_y[:, i] > 0, 0], self.perf_plotter.x[ax_y[:, i] > 0, 1],
                c='r', alpha=0.1, label='Before cleaning'
            )

            # After cleaning bounds
            plt.scatter(
                self.perf_plotter.x[ax_clean_y[:, i] > 0, 0], self.perf_plotter.x[ax_clean_y[:, i] > 0, 1],
                c='b', marker="+", alpha=0.1, label='After cleaning'
            )

            plt.title(f'Component {i}: Cleaning visualisation')
            plt.legend()
            plt.show()