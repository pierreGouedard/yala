# Global imports
from scipy.sparse import csc_matrix, vstack
from numpy.random import binomial
from firing_graph.tools.helpers.servers import ArrayServer
from numpy import array
# Local import


class YalaUnclassifiedServer(ArrayServer):
    def __init__(self, n_parallel, p_dropout, sax_forward, sax_backward, **kwargs):
        self.n_parallel = n_parallel
        self.p_forward_dropout = p_dropout

        # Set sax_mask
        if n_parallel > 1:
            l_indices = sum([[i] * n_parallel for i in range(sax_backward.shape[1])], [])
            sax_backward = sax_backward[:, l_indices]

        self.parallel_mask = csc_matrix(sax_backward.shape, dtype=int)
        if p_dropout > 0:
            self.parallel_mask = csc_matrix(binomial(1, p_dropout, sax_backward.shape).astype(int))

        super(YalaUnclassifiedServer, self).__init__(
            **dict(sax_forward=sax_forward, sax_backward=sax_backward, mask_method="count", **kwargs)
        )

    def update_mask_with_pattern(self, pattern):
        if pattern is None:
            return

        sax_mask = super().propagate_all(pattern, return_activations=False)
        super().update_mask(sax_mask)

    def next_forward(self, n=1, update_step=True):
        l_positions = self.recursive_positions(self.step_forward, n, self.parallel_mask.shape[0])
        sax_parallel_mask = vstack([self.parallel_mask[start:end, :].tocsr() for (start, end) in l_positions])

        super().next_forward(n, update_step)
        # Add to forward mask initial mask
        if self.sax_mask_forward is not None:
            self.sax_mask_forward = (self.sax_mask_forward + sax_parallel_mask > 0).astype(self.dtype_forward)
        else:
            self.sax_mask_forward = sax_parallel_mask

        return self

    def get_init_precision(self, **kwargs):
        return super().get_init_precision(self.parallel_mask)


class YalaMisclassifiedServer(ArrayServer):

    def __init__(self, n_parallel, p_dropout, sax_forward, sax_backward, **kwargs):
        self.n_parallel = n_parallel
        self.p_forward_dropout = p_dropout

        # Set sax_mask
        if n_parallel > 1:
            l_indices = sum([[i] * n_parallel for i in range(sax_backward.shape[1])], [])
            sax_backward = sax_backward[:, l_indices]

        self.parallel_mask = csc_matrix(sax_backward.shape, dtype=int)
        if p_dropout > 0:
            self.parallel_mask = csc_matrix(binomial(1, p_dropout, sax_backward.shape).astype(int))

        super(YalaMisclassifiedServer, self).__init__(
            **dict(sax_forward=sax_forward, sax_backward=sax_backward, mask_method='proba', **kwargs)
        )

    def update_mask_with_pattern(self, pattern):
        if pattern is None:
            return

        ax_precisions = array([p['precision'] for p in sorted(pattern.partitions, key=lambda x: x['indices'][0])])
        sax_mask = super().propagate_all(pattern, ax_values=ax_precisions)
        super().update_mask(sax_mask)

    def next_forward(self, n=1, update_step=True):
        l_positions = self.recursive_positions(self.step_forward, n, self.parallel_mask.shape[0])
        sax_parallel_mask = vstack([self.parallel_mask[start:end, :].tocsr() for (start, end) in l_positions])

        super().next_forward(n, update_step)
        # Add to forward mask initial mask
        if self.sax_mask_forward is not None:
            self.sax_mask_forward = (self.sax_mask_forward + sax_parallel_mask > 0).astype(self.dtype_forward)
        else:
            self.sax_mask_forward = sax_parallel_mask

        return self

    def get_init_precision(self, **kwargs):
        return super().get_init_precision(self.parallel_mask)