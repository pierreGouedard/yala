# Global import
from scipy.sparse import csc_matrix
from matplotlib import pyplot as plt
import numpy as np

# Local import
from src.model.utils.firing_graph import YalaFiringGraph
from src.model.utils.linalg import expand, bounds


class Cleaner:
    """Cleaner"""
    def __init__(
            self, server, bitmap, batch_size, min_bounds=2, perf_plotter=None, plot_perf_enable=False
    ):
        self.server = server
        self.bitmap = bitmap
        self.batch_size = batch_size
        self.min_bound = min_bounds
        self.plot_perf_enable = plot_perf_enable
        self.perf_plotter = perf_plotter

    def clean_component(self, components):
        # Create Mask and exploded firing graph
        msk_comp = components.copy(levels=components.levels - 1)
        msk_fg = YalaFiringGraph.from_fg_comp(msk_comp)

        # Create comps
        sax_mask = expand(components.inputs, self.bitmap, n=2).astype(bool)
        xpld_comp = components.complement(sax_mask).explode(self.bitmap)
        xpld_fg = YalaFiringGraph.from_fg_comp(xpld_comp)

        # Propagate batch size of signal
        sax_msk, sax_xpld = self.propagate_signal(msk_fg, xpld_fg)

        # compute masks
        ax_mask = (sax_msk.multiply(sax_xpld).sum(axis=0) > 0).A.reshape((len(components), self.bitmap.nf))
        sax_bit_msk = self.bitmap.f2b(csc_matrix(ax_mask, dtype=bool).T)

        if self.plot_perf_enable:
            cleaned_components = self.update_comp(components, components.inputs.multiply(sax_bit_msk), inplace=False)
            self.visualize_cleaning(components, cleaned_components)

        return self.update_comp(components, components.inputs.multiply(sax_bit_msk))

    def propagate_signal(self, msk_fg, xpld_fg):
        sax_x = self.server.next_forward(n=self.batch_size, update_step=False).sax_data_forward

        # Explode mask output
        sax_msk = msk_fg.propagate(sax_x)[:, sum([[i] * self.bitmap.nf for i in range(len(msk_fg.partitions))], [])]

        return sax_msk, xpld_fg.propagate(sax_x)

    def update_comp(self, comp, sax_inputs, inplace=True):
        ax_levels = self.bitmap.b2f(sax_inputs.astype(bool)).A.sum(axis=1)
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

    def visualize_cleaning(self, comps, cleaned_comps):
        sax_x = self.server.get_sub_forward(self.perf_plotter.indices)

        ax_y = YalaFiringGraph.from_fg_comp(
            comps.update(levels=np.ones(len(comps)), inputs=bounds(comps.inputs, self.bitmap))
        ).propagate(sax_x).A

        ax_clean_y = YalaFiringGraph.from_fg_comp(
            cleaned_comps.update(levels=np.ones(len(cleaned_comps)), inputs=bounds(cleaned_comps.inputs, self.bitmap))
        ).propagate(sax_x).A

        for i in range(ax_y.shape[1]):
            # Before cleaning bounds
            plt.scatter(
                self.perf_plotter.x[ax_y[:, i] > 0, 0], self.perf_plotter.x[ax_y[:, i] > 0, 1],
                c='r', alpha=0.1, label='Before cleaning'
            )

            # After cleaning bounds
            plt.scatter(
                self.perf_plotter.x[ax_clean_y[:, i] > 0, 0], self.perf_plotter.x[ax_clean_y[:, i] > 0, 1],
                c='b', marker="+", alpha=0.1, label='After cleaning'
            )

            plt.title(f'Component {i}: Cleaning visualisation')
            plt.legend()
            plt.show()
