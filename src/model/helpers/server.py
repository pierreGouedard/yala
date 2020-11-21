# Global imports
from scipy.sparse import vstack, csc_matrix, diags
from numpy import uint8, uint16, vectorize, zeros
from numpy.random import binomial
from firing_graph.tools.helpers.servers import ArrayServer

# Local import


class YalaUnclassifiedServer(ArrayServer):

    def update_mask(self, pattern_mask, **kwargs):
        if pattern_mask is None:
            return

        sax_mask = super().update_mask(pattern_mask)

        if self.sax_mask is not None:
            self.sax_mask += sax_mask.astype(uint16)
        else:
            self.sax_mask = sax_mask.astype(uint16)

    def apply_mask(self, sax_data, type, l_positions):
        if type == 'forward' and self.sax_mask is not None:
            sax_mask = vstack([self.sax_mask[start:end, :].tocsr() for (start, end) in l_positions])
            if self.dropout_rate_mask > 0:
                dropout_func = vectorize(lambda x: binomial(int(x), 1 - self.dropout_rate_mask) > 0 if x > 0 else 0)
                sax_mask.data = dropout_func(sax_mask.data)
                sax_mask.eliminate_zeros()

            sax_data = diags(sax_mask.sum(axis=1).A[:, 0] == 0, dtype=bool).dot(sax_data)

        return sax_data


class YalaMisclassifiedServer(ArrayServer):

    def __init__(self, min_probas, **kwargs):

        super(YalaMisclassifiedServer, self).__init__(**kwargs)
        self.ax_probas = zeros(self.sax_mask.shape)
        self.min_probas = min_probas
        self.threshold = 0.9
        self.ax_counts = zeros(self.sax_mask.shape)

    def update_mask(self, pattern_mask, **kwargs):
        if pattern_mask is None:
            return

        # Get predictions
        ax_precisions = [p['precision'] for p in sorted(pattern_mask.partitions, key=lambda x: x['indices'][0])]
        ax_values = super().update_mask(pattern_mask, ax_values=ax_precisions).A

        # Merge probas
        self.ax_probas = self.ax_probas + ax_values
        self.ax_counts += (ax_values > 0)
        ax_probas = self.ax_probas / (self.ax_counts + 1e-6)

        sax_mask = super().classification_mask(csc_matrix(ax_probas > self.threshold))

        #sax_mask = super().classification_mask(csc_matrix(ax_probas > ax_probas[:, [1, 0]]))

        if self.sax_mask is not None:
            self.sax_mask += sax_mask.astype(uint16)
        else:
            self.sax_mask = sax_mask.astype(uint8)

    def apply_mask(self, sax_data, type, l_positions):
        if type == 'forward' and self.sax_mask is not None:
            sax_mask = vstack([self.sax_mask[start:end, :].tocsr() for (start, end) in l_positions])
            if self.dropout_rate_mask > 0:
                dropout_func = vectorize(lambda x: binomial(int(x), 1 - self.dropout_rate_mask) > 0 if x > 0 else 0)
                sax_mask.data = dropout_func(sax_mask.data)
                sax_mask.eliminate_zeros()

            sax_data = diags(sax_mask.sum(axis=1).A[:, 0] == 0, dtype=bool).dot(sax_data)

        return sax_data
