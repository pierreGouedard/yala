# Global imports
from scipy.sparse import csc_matrix, vstack, diags
from numpy.random import binomial
from firing_graph.tools.helpers.servers import ArrayServer
from numpy import array
# Local import


class YalaUnclassifiedServer(ArrayServer):
    def __init__(self, sax_forward, sax_backward, **kwargs):
        super(YalaUnclassifiedServer, self).__init__(
            **dict(sax_forward=sax_forward, sax_backward=sax_backward, mask_method="count", **kwargs)
        )

    def update_mask_with_pattern(self, pattern):
        if pattern is None:
            return
        sax_mask = super().propagate_all(pattern)
        super().update_mask(sax_mask)

    def next_forward(self, n=1, update_step=True):
        super().next_forward(n, update_step)
        return self

    def next_masked_forward(self, n=1, update_step=True):
        self.next_forward(n, update_step)

        if self.sax_mask_forward.nnz > 0:
            return diags(~(self.sax_mask_forward.A[:, 0] > 0), dtype=bool).dot(self.sax_data_forward)

        return self.sax_data_forward

    def get_init_precision(self, **kwargs):
        return super().get_init_precision()


class YalaMisclassifiedServer(ArrayServer):

    def __init__(self, sax_forward, sax_backward, **kwargs):
        super(YalaMisclassifiedServer, self).__init__(
            **dict(sax_forward=sax_forward, sax_backward=sax_backward, mask_method='proba', **kwargs)
        )

    def update_mask_with_pattern(self, pattern):
        if pattern is None:
            return

        ax_p_dropout = array([max(2 * p['precision'] - 1, 2 * (1 - p['precision']) - 1) for p in pattern.partitions])
        sax_mask = super().propagate_all(pattern, ax_values=ax_p_dropout)
        super().update_mask(sax_mask)

    def next_forward(self, n=1, update_step=True):
        super().next_forward(n, update_step)
        return self

    def next_masked_forward(self, n=1, update_step=True):
        self.next_forward(n, update_step)

        if self.sax_mask_forward.nnz > 0:
            return diags(~(self.sax_mask_forward.A[:, 0] > 0), dtype=bool).dot(self.sax_data_forward)

        return self.sax_data_forward

    def get_init_precision(self, **kwargs):
        return super().get_init_precision()
