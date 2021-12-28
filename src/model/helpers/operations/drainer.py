# Global import
import numpy as np
from dataclasses import asdict
from itertools import groupby
from firing_graph.solver.drainer import FiringGraphDrainer
from scipy.sparse import diags

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
        if self.plot_perf_enabled is None:
            raise ValueError('Impossible to visualize firing graph: not plot perf.')

        # Get masked activations
        sax_x = self.server.get_sub_forward(self.perf_plotter.indices)
        ax_yhat = firing_graph.propagate(sax_x).A

        # Plot perf viz
        self.perf_plotter(ax_yhat)

    def prepare(self, component):
        # Update drainer params
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

    def select_inputs(self, sax_weight, sax_count):

        ax_p, ax_r = self.drainer_params.feedbacks.get_all()
        ax_w, ax_target_prec = self.drainer_params.weights, self.drainer_params.limit_precisions()

        # Get input weights and count
        sax_mask = (sax_weight > 0).multiply(sax_count > 0)

        sax_nom = sax_weight.multiply(sax_mask) - sax_mask.dot(diags(ax_w, format='csc'))
        sax_denom = sax_mask.multiply(sax_count.dot(diags(ax_p + ax_r, format='csc')))
        sax_precision = sax_nom.multiply(sax_denom.astype(float).power(-1))
        sax_precision += (sax_precision != 0).dot(diags(ax_p / (ax_p + ax_r), format='csc'))

        return sax_precision > (sax_precision > 0).dot(diags(ax_target_prec, format='csc'))

    def reset(self):
        self.reset_all()
        self.firing_graph, self.fg_mask = None, None

    def update_drainer_params(self, component):
        # Get masked activations
        sax_x = self.server.next_forward(n=self.drainer_params.batch_size, update_step=False).sax_data_forward
        sax_y = self.server.next_backward(n=self.drainer_params.batch_size, update_step=False).sax_data_backward

        # Compute precision of each vertex and update partitions
        sax_x = YalaBasePatterns.from_fg_comp(component).propagate(sax_x)
        self.drainer_params.precisions = (sax_y.T.astype(int).dot(sax_x) / (sax_x.sum(axis=0) + 1e-6)).A[0]

        # Compute penalty / rewards
        self.drainer_params = init_parameters(self.drainer_params, self.min_firing)
        self.update_pr(**asdict(self.drainer_params.feedbacks))

    def build_patterns(self, component):
        pass
