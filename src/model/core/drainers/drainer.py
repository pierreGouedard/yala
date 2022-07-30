# Global import
from dataclasses import asdict
from firing_graph.solver.drainer import FiringGraphDrainer
from scipy.sparse import diags

# Local import
from src.model.core.drainers.utils import init_parameters
from src.model.utils.firing_graph import YalaFiringGraph, YalaTopPattern
from src.model.utils.spmat_op import add_connex


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
        self.base_component = None

        # call parent constructor
        super().__init__(None, server, self.drainer_params.batch_size)

    def prepare(self, component, mask_component, base_component):
        # Instantiate firing graphs
        self.fg_mask = YalaFiringGraph.from_fg_comp(mask_component)
        self.firing_graph = YalaFiringGraph.from_fg_comp(component)
        self.base_component = base_component

        # Set drainer params & set weights
        self.setup_params()

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

    def select(self, **kwargs):
        # Get active & drained bits
        sax_active_inputs = self.select_inputs(self.firing_graph.Iw, self.firing_graph.backward_firing['i'])

        # Add connex new inputs
        sax_support_bits = add_connex(self.base_component.inputs, sax_active_inputs, self.bitmap)

        # Build component
        self.base_component.update(inputs=sax_support_bits)

        return self.update_partition_metrics()

    def reset(self):
        self.reset_all()
        self.firing_graph, self.fg_mask, self.base_component = None, None, None

    def get_triplet(self,):
        # Get masked activations
        sax_x = self.server.next_forward(n=self.drainer_params.batch_size, update_step=False).sax_data_forward
        sax_y = self.server.next_backward(n=self.drainer_params.batch_size, update_step=False).sax_data_backward

        # propagate through firing graph
        sax_fg = YalaFiringGraph.from_fg_comp(self.base_component).propagate(sax_x)

        return sax_x, sax_y, sax_fg

    def setup_params(self):

        # Get signals to estimate precision
        _, sax_y, sax_fg = self.get_triplet()

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

    def update_partition_metrics(self):
        ax_areas = self.base_component.inputs.sum(axis=0).A[0, :] / \
            (self.bitmap.b2f(self.base_component.inputs.astype(bool)).A.sum(axis=1) + 1e-6)
        return self.base_component.update(
            partitions=[
                {**p, "area": ax_areas[i], "precision": self.drainer_params.precisions[i]}
                for i, p in enumerate(self.base_component.partitions)
            ]
        )