# Global import
import numpy as np
from dataclasses import dataclass
from scipy.sparse import csc_matrix, hstack as sphstack
from typing import Dict

# Local import
from src.model.utils.data_models import FgComponents
from src.model.core.drainers.visualizer import Visualizer
from src.model.utils.linalg import expand, shrink


class Shaper(Visualizer):
    """Shaper"""

    def __init__(
            self, server, bitmap, drainer_params, min_firing=100, min_bounds=2, perf_plotter=None,
            plot_perf_enabled=False, advanced_plot_perf_enabled=False
    ):
        # call parent constructor
        self.mask_bound_manager = None

        super().__init__(
            server, bitmap, drainer_params, min_firing, min_bounds, perf_plotter=perf_plotter,
            plot_perf_enabled=plot_perf_enabled, advanced_plot_perf_enabled=advanced_plot_perf_enabled
        )

    def reset(self):
        self.mask_bound_manager = None
        super().reset()

    def init_var(self, cmpnts):
        mask = {
            i: self.bitmap.bf_map[:, self.bitmap.b2f(cmpnts.inputs > 0).A[i, :]]
            for i in range(len(cmpnts))
        }
        self.mask_bound_manager = MaskBoundManager(cmpnts.levels.copy(), cmpnts.levels.copy(), mask)

        return FgComponents.empty_comp()

    def shape(self, base_components, n_expand=4, p_shrink=0.4):

        conv_components = self.init_var(base_components)
        d_areas = {p['id']: p.get('area', 0) for p in base_components.partitions}
        while len(base_components) > 0:
            # Shrink bound to drain
            import IPython
            IPython.embed()
            sax_drain_inputs = base_components.inputs.multiply(self.mask_bound_manager.get_drain_bmask())
            sax_shrink_inputs = shrink(sax_drain_inputs, self.bitmap, p_shrink)

            drain_components = FgComponents(
                inputs=sax_shrink_inputs, levels=np.ones(len(self.mask_bound_manager)),
                partitions=base_components.partitions
            ).complement(sax_mask=expand(sax_drain_inputs, self.bitmap, n=n_expand))

            # Update base component by replacing current bound with the shrinked one.
            sax_cmplmnt_inputs = base_components.inputs.multiply(self.mask_bound_manager.get_drain_cmplmnt_bmask())

            # Update base components & Create mask component
            base_components.inputs = sax_cmplmnt_inputs + sax_shrink_inputs
            mask_components = base_components.copy(inputs=sax_cmplmnt_inputs, levels=base_components.levels - 1)

            # Drain
            base_components = super().prepare(drain_components, mask_components, base_components) \
                .drain_all() \
                .select()

            if self.advanced_plot_perf_enabled:
                self.visualize_shaping(drain_components, mask_components, base_components)

            self.mask_bound_manager.decrement()
            # TODO: the below line doesn"t work
            conv_components, base_components, d_areas = self.pop_conv_comp(base_components, conv_components, d_areas)

        if self.plot_perf_enabled:
            self.visualize_comp(conv_components)

        return conv_components

    def pop_conv_comp(self, base_components, conv_components, d_areas):
        i, stop = 0, False
        while not stop:
            comp = base_components[i]
            if self.mask_bound_manager.counter[i] == 0:
                if abs(comp.partitions[0]['area'] - d_areas[comp.partitions[0]['id']]) < 2e-1:
                    conv_components += comp
                    base_components.pop(i)
                    self.mask_bound_manager.pop(i)
                    d_areas.pop(comp.partitions[0]['id'])
                else:
                    self.mask_bound_manager.reset(i)
                    d_areas[comp.partitions[0]['id']] = comp.partitions[0]['area']
            else:
                d_areas[comp.partitions[0]['id']] = comp.partitions[0]['area']

            i += 1
            stop = i >= len(base_components)

        import IPython
        IPython.embed()
        return conv_components, base_components, d_areas


@dataclass
class MaskBoundManager:
    counter: np.array
    sizes: np.array
    mask: Dict[int, np.array]

    def __len__(self):
        return self.counter.shape[0]

    def get_drain_fmask(self):
        for i, c in enumerate(self.counter):
            yield i, csc_matrix(np.array(np.eye(self.sizes[i], dtype=bool)[:, [c - 1]]), dtype=bool)

    def get_drain_cmplmnt_fmask(self):
        for i, c in enumerate(self.counter):
            yield i, csc_matrix(np.array(~np.eye(self.sizes[i], dtype=bool)[:, [c - 1]]), dtype=bool)

    def get_drain_bmask(self):
        return sphstack([self.mask[i].dot(ax_mask) for i, ax_mask in self.get_drain_fmask()]).astype(int)

    def get_drain_cmplmnt_bmask(self):
        return sphstack([self.mask[i].dot(ax_mask) for i, ax_mask in self.get_drain_cmplmnt_fmask()]).astype(int)

    def decrement(self):
        self.counter -= 1

    def reset(self, i=None):
        if i is not None:
            self.counter[i] = self.sizes[i]
        else:
            self.counter = self.sizes.copy()

    def pop(self, i):
        # Imitate pop of FG component
        self.counter = np.array([c for j, c in enumerate(self.counter) if j != i])
        self.sizes = np.array([s for j, s in enumerate(self.sizes) if j != i])
        self.mask = np.stack([ax_mask for j, ax_mask in enumerate(self.mask) if j != i])