# Global imports
from numpy import array
from numpy import vectorize
from numpy.random import binomial
from firing_graph.tools.helpers.servers import ArrayServer


# Local import


class YalaUnclassifiedServer(ArrayServer):
    def __init__(self, sax_forward, sax_backward, **kwargs):
        super(YalaUnclassifiedServer, self).__init__(
            **dict(sax_forward=sax_forward, sax_backward=sax_backward, mask_method="count", **kwargs)
        )

    def next_forward(self, n=1, update_step=True):
        super().next_forward(n, update_step)
        return self

    def apply_mask_method(self, ax_mask):
        if self.dropout_rate_mask > 0:
            dropout_func = vectorize(lambda x: binomial(int(x), 1 - self.dropout_rate_mask) < 0 if x > 0 else True)
            ax_mask = dropout_func(ax_mask)
        else:
            ax_mask = ax_mask <= 0

        return ax_mask

    def update_param_mask_with_pattern(self, pattern, update_mask=True):
        if pattern is None:
            return

        ax_mask = super().propagate_all(pattern).A[:, 0]
        self.update_param_mask(ax_mask, update_mask)

    def update_param_mask(self, ax_param_mask, update_mask=True):
        if self.ax_param_mask is not None:
            self.ax_param_mask += ax_param_mask

        else:
            self.ax_param_mask = ax_param_mask

        if update_mask:
            super().update_mask()


class YalaMisclassifiedServer(ArrayServer):

    def __init__(self, sax_forward, sax_backward, **kwargs):
        super(YalaMisclassifiedServer, self).__init__(
            **dict(sax_forward=sax_forward, sax_backward=sax_backward, mask_method='proba', **kwargs)
        )

    def next_forward(self, n=1, update_step=True):
        super().next_forward(n, update_step)
        return self

    def apply_mask_method(self, ax_mask):

        dropout_func = vectorize(lambda x: binomial(1, max(0, min(x, 1))) == 0)
        ax_mask = dropout_func(ax_mask)

        return ax_mask.astype(bool)

    def update_param_mask_with_pattern(self, pattern):
        if pattern is None:
            return

        ax_p_dropout = array([max(2 * p['precision'] - 1, 2 * (1 - p['precision']) - 1) for p in pattern.partitions])
        ax_mask = super().propagate_all(pattern, ax_values=ax_p_dropout).A[:, 0]
        self.update_param_mask(ax_mask)

    def update_param_mask(self, ax_param_mask, update_mask=True):
        if self.ax_param_mask is not None:
            self.ax_param_mask[self.ax_param_mask < ax_param_mask] = (
                    self.ax_param_mask[self.ax_param_mask < ax_param_mask] * 0.8 +
                    ax_param_mask[self.ax_param_mask < ax_param_mask] * 0.2
            )

        else:
            self.ax_param_mask = ax_param_mask

        if update_mask:
            super().update_mask()
