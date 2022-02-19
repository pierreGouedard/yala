# Global import
from dataclasses import asdict
from itertools import groupby
from firing_graph.solver.drainer import FiringGraphDrainer
from scipy.sparse import diags
from abc import abstractmethod
from functools import lru_cache

# Local import
from src.model.helpers.data_models import FgComponents
from src.model.utils import init_parameters
from src.model.helpers.patterns import YalaBasePatterns, YalaTopPattern


class YalaDrainer(FiringGraphDrainer):
    """Abstract child of Firing Graph specific to YALA algorithm."""
    def __init__(self, server, sax_bf_map, drainer_params, min_firing=100, level_delta=0):
        # Map bit to features and candidate features
        self.bf_map = sax_bf_map

        # Parameters for draining
        self.drainer_params = drainer_params

        # complement attributes
        self.min_firing = min_firing
        self.level_delta = level_delta

        # Handy
        self.pre_draining_fg = None

        # call parent constructor
        super().__init__(None, server, self.drainer_params.batch_size)

    def b2f(self, sax_x):
        return sax_x.T.dot(self.bf_map)

    def f2b(self, sax_x):
        return self.bf_map.dot(sax_x)

    @lru_cache()
    def get_card_features(self, n_repeat):
        return self.bf_map.sum(axis=0).A[[0] * n_repeat, :]

    def prepare(self, component):
        # Get triplet signals
        sax_x, sax_y, sax_fg = self.get_triplet(component)

        # Build top and bottom patterns
        self.build_patterns(component, **{"x": sax_x, "y": sax_y, "fg": sax_fg})

        # Set drainer params & set weights
        self.setup_params(sax_y, sax_fg)

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
        # Compute new inputs, levels and partitions
        sax_inputs = self.select_support_bits(self.firing_graph.Iw, self.firing_graph.backward_firing['i'])
        l_partitions = self.update_partition_metrics(sax_inputs)

        # Create component
        fg_comp = FgComponents(inputs=sax_inputs, partitions=l_partitions, levels=self.b2f(sax_inputs).A.sum(axis=1))

        return fg_comp

    @abstractmethod
    def select_support_bits(self, sax_drained_weights, sax_count_activations):
        pass

    def update_partition_metrics(self, sax_inputs):
        # Compute metrics
        ax_areas = sax_inputs.sum(axis=0).A[0, :] / self.b2f(sax_inputs).A.sum(axis=1)

        l_metrics = [
            {**self.pre_draining_fg.partitions[i], "precision": self.drainer_params.precisions[i], "area": ax_areas[i]}
            for i in range(sax_inputs.shape[1])
        ]
        return l_metrics

    def select_inputs(self, sax_weight, sax_count):

        ax_p, ax_r = self.drainer_params.feedbacks.get_all()
        ax_w, ax_target_prec = self.drainer_params.weights, self.drainer_params.limit_precisions()

        # Get input weights and count
        sax_mask = (sax_weight > 0).multiply(sax_count > 0)

        sax_nom = sax_weight.multiply(sax_mask) - sax_mask.dot(diags(ax_w, format='csc'))
        sax_denom = sax_mask.multiply(sax_count.dot(diags(ax_p + ax_r, format='csc')))
        sax_precision = sax_nom.multiply(sax_denom.astype(float).power(-1))
        sax_precision += (sax_precision != 0).dot(diags(ax_p / (ax_p + ax_r), format='csc'))

        # Compute selected inputs
        return sax_precision > (sax_precision > 0).dot(diags(ax_target_prec, format='csc'))

    def reset(self):
        self.reset_all()
        self.firing_graph, self.fg_mask = None, None

    def get_triplet(self, component):
        # Get masked activations
        sax_x = self.server.next_forward(n=self.drainer_params.batch_size, update_step=False).sax_data_forward
        sax_y = self.server.next_backward(n=self.drainer_params.batch_size, update_step=False).sax_data_backward

        # propagate through firing graph
        sax_fg = YalaBasePatterns.from_fg_comp(component.copy(levels=component.levels - self.level_delta))\
            .propagate(sax_x)

        return sax_x, sax_y, sax_fg

    def setup_params(self, sax_y, sax_fg):
        self.drainer_params.precisions = (sax_y.T.astype(int).dot(sax_fg) / (sax_fg.sum(axis=0) + 1e-6)).A[0]

        # Compute penalty / rewards
        self.drainer_params = init_parameters(self.drainer_params, self.min_firing)
        self.update_pr(**asdict(self.drainer_params.feedbacks))

        # Update base matrix input's weights
        sax_weights = diags(self.drainer_params.weights, format='csc', dtype=self.firing_graph.matrices['Iw'].dtype)
        self.firing_graph.matrices['Iw'] = self.firing_graph.matrices['Iw'].dot(sax_weights)

        # Update mask draining
        self.firing_graph.matrices['Im'] = self.firing_graph.I

        # Update firing graph from parent
        self.reset_all()

    @abstractmethod
    def build_patterns(self, component, **kwargs):
        pass
