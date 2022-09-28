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
    def __init__(self, server, drainer_params):
        # Map bit to features and candidate features

        # Parameters for draining
        self.params = drainer_params
        self.ax_pr, self.ax_w = None, None

        # call parent constructor
        super().__init__(None, server, self.params.batch_size)

    def select(self, base_mask, other_mask):
        # Get support bits
        sax_support_bits = clip_inputs(
            self.firing_graph.Iw, self.firing_graph.backward_firing, self.ax_w, self.ax_p, self.ax_r,
            self.params.limit_precisions(self.ax_pr)
        )

        # Update firing graph inputs
        sax_support_bits = add_connex(base_mask, sax_support_bits, self.server.bitmap)
        self.firing_graph.matrices['Iw'] = self.firing_graph.matrices['Iw'].multiply(other_mask) + sax_support_bits
        self.firing_graph.refresh_matrices()

        return self.firing_graph.to_comp()

    def drain(self, n=1):
        # Compute penalty / rewards / weight and update firing graph
        self.firing_graph.matrices['Iw'] = (
            self.firing_graph.matrices['Iw']
            .dot(diags(self.ax_w, format='csr', dtype=int32))
            .tocsr()
        )

        # reset signals
        self.reset(server=True)

        return super().drain()

    def setup_params(self, comps):
        # Set drainer params
        self.ax_pr = array([p['precision'] for p in comps.partitions])
        self.ax_p, self.ax_r, self.ax_w = set_drain_params(self.ax_pr, self.params.margin)
        comps.partitions = [
            {**d, 'p': self.ax_p[i], 'r': self.ax_r[i], 'w': self.ax_w[i]}
            for i, d in enumerate(comps.partitions)
        ]
        # Set server backward pattern
        self.server.update_pattern_backward(YalaTopPattern(
            self.server.n_label, [p['label_id'] for p in comps.partitions]
        ))

    def update_params(self, comps):
        # Update drainer params
        self.ax_p = array([d['p'] for d in comps.partitions])
        self.ax_r = array([d['r'] for d in comps.partitions])
        self.ax_w = array([d['w'] for d in comps.partitions], dtype=int32)
        self.ax_pr = array([p['precision'] for p in comps.partitions])

        # Update server backward pattern
        self.server.update_pattern_backward(YalaTopPattern(
            self.server.n_label, [p['label_id'] for p in comps.partitions]
        ))





