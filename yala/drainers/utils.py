# Global import
import numpy as np
from dataclasses import dataclass
from scipy.sparse import hstack as sphstack

# Local import
from yala.utils.data_models import FgComponents


@dataclass
class MaskManager:
    counter: np.array
    sizes: np.array
    changes: np.array
    mask: np.ndarray

    def __init__(self, comps, bitmap):
        self.mask = np.stack([bitmap.bf_map[:, bitmap.b2f(comp.inputs).A[0, :]] for comp in comps])
        self.sizes, self.counter = comps.levels.copy(), comps.levels.copy()
        self.changes = np.zeros(len(self), dtype=bool)

    def __len__(self):
        return self.counter.shape[0]

    def get_curr_bmask(self, sax_inputs):
        sax_curr_mask = sphstack([self.mask[i][:, c - 1] for i, c in enumerate(self.counter)])
        return sax_inputs.multiply(sax_curr_mask)

    def get_oth_bmask(self, sax_inputs):
        sax_oth_mask = sphstack([
            sum([self.mask[i][:, j] for j in range(self.sizes[i]) if j != c - 1]) for i, c in enumerate(self.counter)
        ])

        return sax_inputs.multiply(sax_oth_mask)

    def update_counter(self, sax_old_inputs, sax_new_inputs):
        # Build mask of unchanged inputs
        ax_mask = sax_old_inputs.sum(axis=0).A[0] != self.get_curr_bmask(sax_new_inputs).sum(axis=0).A[0]

        # Track changes and decrement counter with no changes
        self.changes += ax_mask
        self.counter -= 1

        return self

    def reset(self, i=None):
        if i is not None:
            self.counter[i] = self.sizes[i]
            self.changes[i] = False
        else:
            self.counter = self.sizes.copy()
            self.changes = np.zeros(len(self))

    def pop_no_changes(self, comps):

        i, stop, conv_comps = 0, False, FgComponents.empty_comp()
        while not stop:
            if self.counter[i] == 0:
                if self.changes[i]:
                    self.reset(i)
                    i += 1

                else:
                    conv_comps += comps.pop(i)
                    self._pop(i)
            else:
                i += 1

            # Increment i
            stop = i >= len(comps)

        return conv_comps, comps

    def _pop(self, i):
        # Imitate pop of FG component
        self.counter = np.array([c for j, c in enumerate(self.counter) if j != i])
        self.sizes = np.array([s for j, s in enumerate(self.sizes) if j != i])
        self.changes = np.array([c for j, c in enumerate(self.changes) if j != i])
        l_masks, self.mask = [ax_mask for j, ax_mask in enumerate(self.mask) if j != i], np.array([])
        if l_masks:
            self.mask = np.stack(l_masks)