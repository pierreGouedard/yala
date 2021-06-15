# Global import
import numpy as np
from dataclasses import asdict
from itertools import groupby
from firing_graph.solver.drainer import FiringGraphDrainer

# Local import
from src.model.utils import init_parameters
from src.model.helpers.patterns import YalaBasePatterns, YalaTopPattern


class Refiner(FiringGraphDrainer):
    def __init__(
            self, server, sax_bf_map, drainer_params, min_firing=100, n_update=10
    ):
        # map bit to features
        self.bf_map = sax_bf_map

        # Parameters for draining
        self.drainer_params = drainer_params

        # complement attributes
        self.min_firing, self.n_update = min_firing, n_update

        #
        self.bottom_pattern, self.top_pattern, self.drained_pattern = None, None, None

        super().__init__(
            self.bottom_pattern, server, self.drainer_params.batch_size, **asdict(self.drainer_params.feedbacks)
        )

    def prepare(self, component, ):
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
        self.server.pattern_top = self.top_pattern

        # Drain
        super().drain_all(n_max=self.drainer_params.total_size)

        # Reset top pattern of server
        self.server.pattern_backward, self.server.pattern_top = None, None

    def select(self):
        # For each drained vertex, choose which bounds / features shall be chosen to get to the next step
        # TODO
        pass

    def reset(self):
        self.reset_all()
        self.bottom_pattern, self.top_pattern, self.drained_pattern = None, None, None
        self.firing_graph = None

    def update_precision(self, component):

        # Get masked activations
        sax_x = self.server.next_masked_forward(n=self.drainer_params.batch_size, update_step=False)
        sax_y = self.server.next_backward(n=self.drainer_params.batch_size, update_step=False).sax_data_backward

        # Compute precision of each vertex and update partitions
        sax_x = YalaBasePatterns.from_fg_comp(component).propagate(sax_x)
        ax_precisions = (sax_y.T.astype(int).dot(sax_x) / (sax_x.sum(axis=0) + 1e-6)).A[0]
        component.partitions = [{**p, 'precision': ax_precisions[i]} for i, p in enumerate(component.partitions)]

    def update_drainer_params(self, component):

        # Get precision from component
        ax_precisions = np.array([p['precision'] for p in component.partitions])

        # Compute and update feedbacks and weights used in draining
        feedbacks, weights = init_parameters(ax_precisions, self.drainer_params.margin, self.min_firing)
        self.drainer_params.feedbacks, self.drainer_params.weights = feedbacks, weights

    def build_patterns(self, component):

        # Create top pattern from comp
        self.top_pattern = YalaBasePatterns.from_fg_comp(component)

        # Create base bottom comp
        self.bottom_pattern = YalaBasePatterns.from_fg_comp(component)

        # Augment with unexplored candidate features
        # TODO

        # Update matrices
        self.bottom_pattern.levels = np.ones(len(component))
        self.bottom_pattern.matrices['Iw'] = self.bottom_pattern.matrices['Iw'] * self.drainer_params.weights[0]
        self.bottom_pattern.matrices['Im'] = self.bottom_pattern.I

        # Update firing graph from parent
        self.firing_graph = self.bottom_pattern
        self.reset_all()
