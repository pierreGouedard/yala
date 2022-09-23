# Global import
from firing_graph.drainer import FiringGraphDrainer
from scipy.sparse import diags
from numpy import int32, array

# Local import
from firing_graph.linalg.measure import set_drain_params, clip_inputs
from yala.firing_graph import YalaTopPattern
from yala.linalg.spmat_op import add_connex


class YalaDrainer(FiringGraphDrainer):
    """Abstract child of Firing Graph Drainer specific to YALA algorithm."""
    def __init__(self, server, bitmap, drainer_params):
        # Map bit to features and candidate features
        self.bitmap = bitmap

        # Parameters for draining
        self.params = drainer_params

        # call parent constructor
        super().__init__(None, server, self.params.batch_size)

    def select(self, base_mask, other_mask):
        # Get support bits
        sax_support_bits = clip_inputs(
            self.firing_graph.Iw, self.firing_graph.backward_firing, self.params.weights, self.ax_p, self.ax_r,
            self.params.limit_precisions()
        )

        # Update firing graph inputs
        sax_support_bits = add_connex(base_mask, sax_support_bits, self.bitmap)
        self.firing_graph.matrices['Iw'] = self.firing_graph.matrices['Iw'].multiply(other_mask) + sax_support_bits
        self.firing_graph.refresh_matrices()

        return self.firing_graph.to_comp()

    def drain_all(self, n_max=10000):
        # The below needs to be done at each iteration.
        self.server.pattern_backward = YalaTopPattern(
            self.server.n_label, [p['label_id'] for p in self.firing_graph.partitions]
        )

        # Compute penalty / rewards / weight and update firing graph
        self.firing_graph.matrices['Iw'] = (
            self.firing_graph.matrices['Iw']
            .dot(diags(self.params.weights, format='csr', dtype=int32))
            .tocsr()
        )

        # reset signals
        self.reset(server=True)

        return super().drain_all(n_max)

    def setup_params(self, comps):
        self.params.precisions = array([p['precision'] for p in comps.partitions])
        self.ax_p, self.ax_r, self.params.weights = set_drain_params(self.params.precisions, self.params.margin)




