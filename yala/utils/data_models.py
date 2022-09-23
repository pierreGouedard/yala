"""All core that specify data structures."""
# Global import
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from scipy.sparse import spmatrix, csr_matrix, hstack as sphstack
from copy import deepcopy as copy
from functools import lru_cache
import numpy as np

# Local import


@dataclass
class BitMap:
    bf_map: spmatrix
    nb: int
    nf: int

    def __len__(self):
        return self.bf_map.shape[1]

    def __iter__(self):
        self.__idx = 0
        return self

    def __next__(self):
        if self.__idx >= len(self):
            raise StopIteration

        _next = self[self.__idx]
        self.__idx += 1

        return _next

    def __getitem__(self, i):
        assert isinstance(i, int), 'index should be an integer'
        return self.bf_map[:, i]

    @lru_cache()
    def feature_card(self, n_repeat):
        return self.bf_map.sum(axis=0).A[[0] * n_repeat, :]

    def b2f(self, sax_x):
        return sax_x.T.dot(self.bf_map)

    def f2b(self, sax_x):
        return self.bf_map.dot(sax_x)

    def bitmask(self, sax_x):
        return self.f2b(self.b2f(sax_x).T)


@dataclass
class FgComponents:
    inputs: spmatrix
    mask_inputs: spmatrix
    partitions: List[Dict[str, Any]]
    levels: np.array
    __idx: int = 0

    @staticmethod
    def empty_comp():
        return FgComponents(csr_matrix((0, 0)), csr_matrix((0, 0)), [], np.empty((0,)))

    @property
    def empty(self):
        return len(self.levels) == 0

    def __iter__(self):
        self.__idx = 0
        return self

    def __next__(self):
        if self.__idx >= len(self):
            raise StopIteration

        _next = self[self.__idx]
        self.__idx += 1

        return _next

    def __len__(self):
        return self.levels.shape[0]

    def __getitem__(self, i):
        assert isinstance(i, int), 'index should be an integer'
        return FgComponents(
            inputs=self.inputs[:, i], mask_inputs= self.mask_inputs[:, i],
            partitions=[self.partitions[i]], levels=self.levels[[i]]
        )

    def __add__(self, other):
        if self.inputs.shape[1] == 0:
            sax_inputs = other.inputs
            sax_mask_input = other.mask_inputs
        else:
            sax_inputs = sphstack([self.inputs, other.inputs], format='csr')
            sax_mask_input = sphstack([self.mask_inputs, other.mask_inputs], format='csr')

        return FgComponents(
            inputs=sax_inputs, mask_inputs=sax_mask_input, partitions=self.partitions + other.partitions,
            levels=np.hstack([self.levels, other.levels])
        )

    def update(self, **kwargs):
        if kwargs.get('partitions', None) is not None:
            self.partitions = kwargs['partitions']

        if kwargs.get('levels', None) is not None:
            self.levels = kwargs['levels']

        if kwargs.get('inputs', None) is not None:
            self.inputs = kwargs['inputs']

        if kwargs.get('mask_inputs', None) is not None:
            self.mask_inputs = kwargs['mask_inputs']

        return self

    # TODO: the 2 below are may be not supposed to be here.
    def complement(self, sax_mask=None, inplace=False, **kwargs):
        sax_inputs = csr_matrix((self.inputs > 0).A ^ np.ones(self.inputs.shape, dtype=bool))
        if sax_mask is not None:
            sax_inputs = sax_inputs.multiply(sax_mask)

        if inplace:
            self.update(inputs=sax_inputs, **kwargs)
        else:
            return self.copy(inputs=sax_inputs, **kwargs)

    def pop(self, ind):
        tmp = self[ind]
        l_idx = list(set(range(len(self))).difference({ind}))

        # Update data
        self.partitions = [self.partitions[i] for i in l_idx]
        self.inputs, self.mask_inputs = self.inputs[:, l_idx].tocsr(), self.mask_inputs[:, l_idx].tocsr()
        self.levels = self.levels[l_idx]

        return tmp

    def copy(self, **kwargs):
        return FgComponents(**{
            'inputs': self.inputs.copy(), 'mask_inputs': self.mask_inputs.copy(), 'partitions': copy(self.partitions),
            'levels': self.levels.copy(), **kwargs
        })


@dataclass
class ConvexHullProba:
    counts: Optional[Dict[str, np.array]] = None

    def get_probas(self, comp: FgComponents, bitmap: BitMap):
        ax_support_bounds = bitmap.b2f(comp.inputs).A

        if self.counts is None:
            ax_counts = np.ones(ax_support_bounds.shape)
        else:
            ax_counts = np.vstack([self.counts.get(d['id'], np.ones(bitmap.nf)) for d in comp.partitions])

        ax_p = ~ax_support_bounds / ax_counts

        return ax_p / ax_p.sum(axis=1)[:, np.newaxis]

    def add(self, comp: FgComponents, bitmap: BitMap):
        if self.counts is None:
            self.counts = {}

        ax_counts = bitmap.b2f(comp.inputs).A
        self.counts = {
            d['id']: self.counts.get(d['id'], np.ones(bitmap.nf)) + ax_counts[i].astype(int)
            for i, d in enumerate(comp.partitions)
        }
        return self


@dataclass
class DrainerParameters:
    total_size: int
    batch_size: int
    margin: float
    weights: Optional[np.array] = None
    precisions: Optional[np.array] = None

    def limit_precisions(self):
        return (self.precisions - self.margin).clip(min=self.margin + 0.01)


@dataclass
class TrackerParameters:
    min_delta_area: float
    min_area: float
    max_no_changes: int


