# Global import
from scipy.sparse import csr_matrix
import numpy as np

# Local import
from .firing_graph import YalaFiringGraph
from yala.linalg.spmat_op import expand, explode
from yala.utils.visual import Visualizer


class Cleaner:
    """Cleaner -- Is a class really necessary?"""
    def __init__(
            self, server, batch_size, min_bounds=2, perf_plotter=None, plot_perf_enable=False
    ):
        self.server = server
        self.batch_size = batch_size
        self.min_bound = min_bounds
        self.plot_perf_enable = plot_perf_enable
        self.visualizer = Visualizer(perf_plotter)

    def clean_component(self, components):
        # TODO: there may exists a way faster method: take [d * (d-1) / 2] intersection (2 | d (level=2)) if a bound is
        #   support there is at least one intersection that occurs outside of the bounds of this feature,
        #   if a bound is not support, no intersect of all other bound occur inside the bound of ths feature.
        #   This new method enables to limit the complexity of this method.
        #   Indeed now we just need to propagate on the same number of vertices on inputs that has only there bounds
        #   active. sax_fg.dot(sax_x) => (n vertices * n_inputs) then do the logic with components.inputs.
        #   Problem> if all bounds are support (all other intersection occurs inside the bound tested)
        # Create Mask and exploded firing graph
        msk_comp = components.copy(levels=components.levels - 1)
        msk_fg = YalaFiringGraph.from_comp(msk_comp)

        # Explode vertices bounds
        sax_xpld_inputs, xpld_partitions = explode(
            expand(components.inputs, self.server.bitmap, n=2, keep_only_expanded=True),
            self.server.bitmap, components.partitions
        )
        xpld_fg = YalaFiringGraph.from_inputs(
            sax_xpld_inputs, sax_xpld_inputs, np.ones(sax_xpld_inputs.shape[1]), xpld_partitions
        )

        # Propagate batch size of signal
        sax_msk, sax_xpld = self.propagate_signal(msk_fg, xpld_fg)

        # compute masks
        ax_mask = (sax_msk.multiply(sax_xpld).sum(axis=0) > 0).A.reshape((len(components), self.server.bitmap.nf))
        sax_bit_msk = self.server.bitmap.f2b(csr_matrix(ax_mask, dtype=bool).T)

        if self.plot_perf_enable:
            cleaned_components = self.update_comp(components, components.inputs.multiply(sax_bit_msk), inplace=False)
            self.visualizer.visualize_cleaning(self.server, components, cleaned_components.copy())
            return cleaned_components

        return self.update_comp(components, components.inputs.multiply(sax_bit_msk))

    def propagate_signal(self, msk_fg, xpld_fg):
        sax_x = self.server.next_forward(n=self.batch_size, update_step=False).sax_data_forward

        # Explode mask output
        # TODO: this operation is very memory intensive. (*nf the size that may be very large)
        l_indices = sum([[i] * self.server.bitmap.nf for i in range(len(msk_fg.partitions))], [])
        sax_msk = msk_fg.seq_propagate(sax_x)[:, l_indices]

        return sax_msk, xpld_fg.seq_propagate(sax_x)

    def update_comp(self, comp, sax_inputs, inplace=True):
        ax_levels = self.server.bitmap.b2f(sax_inputs).A.sum(axis=1)
        ax_areas = sax_inputs.sum(axis=0).A[0, :] / (ax_levels + 1e-6)

        if inplace:
            return comp.update(
                partitions=[{**p, "area": ax_areas[i]} for i, p in enumerate(comp.partitions)],
                levels=ax_levels, inputs=sax_inputs
            )
        return comp.copy(
                partitions=[{**p, "area": ax_areas[i]} for i, p in enumerate(comp.partitions)],
                levels=ax_levels, inputs=sax_inputs
            )