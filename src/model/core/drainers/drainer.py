# Global import
from dataclasses import asdict
from firing_graph.solver.drainer import FiringGraphDrainer
from scipy.sparse import diags, csc_matrix
from abc import abstractmethod


# Local import
from src.model.core.data_models import FgComponents
from src.model.utils import init_parameters
from src.model.core.firing_graph import YalaFiringGraph, YalaTopPattern


class YalaDrainer(FiringGraphDrainer):
    """Abstract child of Firing Graph specific to YALA algorithm."""
    def __init__(self, server, bitmap, drainer_params, min_firing=100, min_bounds=2):
        # Map bit to features and candidate features
        self.bitmap = bitmap

        # Parameters for draining
        self.drainer_params = drainer_params

        # complement attributes
        self.min_firing = min_firing
        self.min_bounds = min_bounds

        # Handy
        self.original_inputs = None

        # call parent constructor
        super().__init__(None, server, self.drainer_params.batch_size)

    def prepare(self, component, mask_component, **kwargs):
        # Build top and bottom patterns
        self.build_patterns(component, mask_component, **kwargs)

        # Set drainer params & set weights
        self.setup_params(component)

        return self

    def drain_all(self, **kwargs):
        # Update top and backward pattern of server
        self.server.pattern_backward = YalaTopPattern(
            self.server.n_label, [p['label_id'] for p in self.firing_graph.partitions]
        )

        # Drain
        super().drain_all(n_max=self.drainer_params.total_size)

        # Reset top pattern of server
        self.server.pattern_backward = None

        return self

    def select(self, merge=False):
        # Compute new inputs, levels and partitions
        # TODO: get component drained + mask (or merge)
        sax_inputs = self.select_support_bits(
            self.firing_graph.Iw, self.firing_graph.backward_firing['i']
        )
        l_partitions = self.update_partition_metrics(sax_inputs)

        # Create component
        fg_comp = FgComponents(
            inputs=sax_inputs, partitions=l_partitions, levels=self.bitmap.b2f(sax_inputs).A.sum(axis=1)
        )

        return fg_comp

    def reset(self):
        self.reset_all()
        self.firing_graph, self.fg_mask, self.original_inputs = None, None, None

    def build_patterns(self, component, mask_component, **kwargs):
        # Instantiate mask firing graph
        self.fg_mask = YalaFiringGraph.from_fg_comp(mask_component)

        # Get hull of base
        ch_comp = self.fg_mask.get_convex_hull(
            self.server, self.drainer_params.batch_size,
            mask=self.bitmap.f2b(self.bitmap.b2f(component.inputs.astype(bool)).T)
        )

        # Get firing graph to drain
        self.firing_graph = YalaFiringGraph.from_fg_comp(component.copy(
            inputs=csc_matrix((component.inputs.A ^ ch_comp.inputs.A))
        ))
        self.original_inputs = component.inputs.copy()

    def get_triplet(self, component):
        # Get masked activations
        sax_x = self.server.next_forward(n=self.drainer_params.batch_size, update_step=False).sax_data_forward
        sax_y = self.server.next_backward(n=self.drainer_params.batch_size, update_step=False).sax_data_backward

        # propagate through firing graph
        sax_fg = YalaFiringGraph.from_fg_comp(component.copy(levels=component.levels))\
            .propagate(sax_x)

        return sax_x, sax_y, sax_fg

    def setup_params(self, component):
        # Get signals to estimate precision
        _, sax_y, sax_fg = self.get_triplet(component)

        # Get arg max as label, keep max precision
        ax_precisions = (sax_y.T.astype(int).dot(sax_fg) / (sax_fg.sum(axis=0) + 1e-6)).A
        ax_labels = ax_precisions.argmax(axis=0)
        self.drainer_params.precisions = ax_precisions.max(axis=0)

        # Compute penalty / rewards
        self.drainer_params = init_parameters(self.drainer_params, self.min_firing)
        self.update_pr(**asdict(self.drainer_params.feedbacks))

        # Update base matrix input's weights
        sax_weights = diags(self.drainer_params.weights, format='csc', dtype=self.firing_graph.matrices['Iw'].dtype)
        self.firing_graph.matrices['Iw'] = self.firing_graph.matrices['Iw'].dot(sax_weights)

        # Update mask draining
        self.firing_graph.matrices['Im'] = self.firing_graph.I

        # Update labels
        self.firing_graph.partitions = [
            {**d, 'label_id': ax_labels[i]} for i, d in enumerate(self.firing_graph.partitions)
        ]

        # Update firing graph from parent
        self.reset_all()

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

    @abstractmethod
    def select_support_bits(self, sax_drained_weights, sax_count_activations):
        pass

    def update_partition_metrics(self, sax_inputs):
        # Compute metrics
        ax_areas = sax_inputs.sum(axis=0).A[0, :] / (self.bitmap.b2f(sax_inputs).A.sum(axis=1) + 1e-6)

        l_metrics = [
            {**self.firing_graph.partitions[i], "precision": self.drainer_params.precisions[i], "area": ax_areas[i]}
            for i in range(sax_inputs.shape[1])
        ]
        return l_metrics

