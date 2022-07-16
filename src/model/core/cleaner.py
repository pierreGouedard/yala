# Global import
from scipy.sparse import csc_matrix
from scipy.signal import convolve2d
import numpy as np
# Local import
from src.model.core.firing_graph import YalaFiringGraph
from src.model.core.linalg import expand


class Cleaner:
    """Cleaner"""
    def __init__(self, server, bitmap, batch_size, min_bounds=2):
        self.server = server
        self.bitmap = bitmap
        self.batch_size = batch_size
        self.min_bound = min_bounds

    def clean_component(self, component):
        # Create Mask and exploded firing graph
        msk_component = component.copy(levels=component.levels - 1)
        msk_fg = YalaFiringGraph.from_fg_comp(msk_component)

        # Create comps
        sax_mask = expand(component.inputs, self.bitmap, n=2).astype(bool)
        xpld_comp = component.complement(sax_mask).explode(self.bitmap)
        xpld_fg = YalaFiringGraph.from_fg_comp(xpld_comp)

        # Propagate batch size of signal
        sax_msk, sax_xpld = self.propagate_signal(msk_fg, xpld_fg)

        # compute masks
        ax_mask = (sax_msk.multiply(sax_xpld).sum(axis=0) > 0).A.reshape((len(component), self.bitmap.nf))
        sax_bit_msk = self.bitmap.f2b(csc_matrix(ax_mask, dtype=bool).T)

        return self.update_comp(component, component.inputs.multiply(sax_bit_msk))

    def propagate_signal(self, msk_fg, xpld_fg):
        sax_x = self.server.next_forward(n=self.batch_size, update_step=False).sax_data_forward

        # Explode mask output
        sax_msk = msk_fg.propagate(sax_x)[:, sum([[i] * self.bitmap.nf for i in range(len(msk_fg.partitions))], [])]

        return sax_msk, xpld_fg.propagate(sax_x)

    def update_comp(self, comp, sax_inputs):
        ax_levels = self.bitmap.b2f(sax_inputs.astype(bool)).A.sum(axis=1)
        ax_areas = sax_inputs.sum(axis=0).A[0, :] / (ax_levels + 1e-6)
        return comp.update(
            partitions=[{**p, "area": ax_areas[i]} for i, p in enumerate(comp.partitions)],
            levels=ax_levels, inputs=sax_inputs
        )
