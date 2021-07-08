# Global import
import numpy as np
from dataclasses import asdict
from itertools import groupby
from firing_graph.solver.drainer import FiringGraphDrainer

# Local import
from src.model.utils import init_parameters
from src.model.helpers.patterns import YalaBasePatterns, YalaTopPattern


class YalaDrainer(FiringGraphDrainer):
    """Abstract child of Firing Graph specific to YALA algorithm."""

    def __init__(self, server, sax_bf_map, drainer_params, min_firing=100, perf_plotter=None, plot_perf_enabled=False):
        # map bit to features and candidate features
        self.bf_map, self.mask_candidate = sax_bf_map, None

        # Parameters for draining
        self.drainer_params = drainer_params

        # complement attributes
        self.min_firing = min_firing

        # Optional callable to visualize perf
        self.perf_plotter = perf_plotter
        self.plot_perf_enabled = plot_perf_enabled

        # call parent constructor
        super().__init__(None, server, self.drainer_params.batch_size)

    def visualize_fg(self, firing_graph):
        if self.plot_perf_enabled is None or not self.plot_perf_enabled:
            raise ValueError('Impossible to visualize firing graph: not plot perf or nt enables.')

        # Get masked activations
        sax_x = self.server.get_sub_forward(self.perf_plotter.indices)
        ax_yhat = firing_graph.propagate(sax_x).A

        # Plot perf viz
        self.perf_plotter(ax_yhat)

    def prepare(self, component):
        # Update precision and drainer params
        self.update_precision(component)
        self.update_drainer_params(component)

        # Build top and bottom patterns
        self.build_patterns(component)

        return self

    def drain_all(self, **kwargs):
        # Update top and backward pattern of server
        gr = groupby([(p['label_id'], i) for i, p in enumerate(self.firing_graph.partitions)], key=lambda t: t[0])
        self.server.pattern_backward = YalaTopPattern.from_mapping({k: list(map(lambda x: x[1], v)) for k, v in gr})

        # Drain
        super().drain_all(n_max=self.drainer_params.total_size)

        # Reset top pattern of server
        self.server.pattern_backward = None

        return self

    def select(self):
        pass

    def reset(self):
        self.reset_all()
        self.firing_graph, self.fg_mask = None, None

    def update_precision(self, component):
        # Get masked activations
        sax_x = self.server.next_forward(n=self.drainer_params.batch_size, update_step=False).sax_data_forward
        sax_y = self.server.next_backward(n=self.drainer_params.batch_size, update_step=False).sax_data_backward

        # Compute precision of each vertex and update partitions
        sax_x = YalaBasePatterns.from_fg_comp(component).propagate(sax_x)

        ax_precisions = (sax_y.T.astype(int).dot(sax_x) / (sax_x.sum(axis=0) + 1e-6)).A[0]
        component.partitions = [{**p, 'precision': ax_precisions[i]} for i, p in enumerate(component.partitions)]
        self.drainer_params.precisions = ax_precisions

    def update_drainer_params(self, component):

        # Get precision from component
        ax_precisions = np.array([p['precision'] for p in component.partitions])

        # Compute and update feedbacks and weights used in draining
        self.drainer_params.feedbacks, self.drainer_params.weights = \
            init_parameters(ax_precisions, self.drainer_params.margin, self.min_firing)
        self.update_pr(**asdict(self.drainer_params.feedbacks))

    def build_patterns(self, component):
        pass
