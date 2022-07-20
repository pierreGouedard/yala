"""All classes that specify data structures."""
# Global import
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from scipy.sparse import spmatrix, csc_matrix, hstack as sphstack
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

    def explode(self, sax_x, ind=0):
        return sax_x[:, [ind] * self.nf].multiply(self.bf_map)


@dataclass
class FgComponents:
    inputs: spmatrix
    partitions: List[Dict[str, Any]]
    levels: np.array
    __idx: int = 0

    @staticmethod
    def empty_comp():
        return FgComponents(csc_matrix((0, 0)), [], np.empty((0,)))

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
            inputs=self.inputs[:, i], partitions=[self.partitions[i]], levels=self.levels[[i]]
        )

    def __add__(self, other):
        if self.inputs.shape[1] == 0:
            sax_inputs = other.inputs
        else:
            sax_inputs = sphstack([self.inputs, other.inputs])

        return FgComponents(
            inputs=sax_inputs, partitions=self.partitions + other.partitions,
            levels=np.hstack([self.levels, other.levels])
        )

    def update(self, **kwargs):
        if kwargs.get('partitions', None) is not None:
            self.partitions = kwargs['partitions']

        if kwargs.get('levels', None) is not None:
            self.levels = kwargs['levels']

        if kwargs.get('inputs', None) is not None:
            self.inputs = kwargs['inputs']

        return self

    def complement(self, sax_mask=None, inplace=False, **kwargs):
        sax_inputs = csc_matrix((self.inputs > 0).A ^ np.ones(self.inputs.shape, dtype=bool))
        if sax_mask is not None:
            sax_inputs = sax_inputs.multiply(sax_mask)

        if inplace:
            self.update(inputs=sax_inputs, **kwargs)
        else:
            return self.copy(inputs=sax_inputs, **kwargs)

    def explode(self, bitmap: BitMap, inplace=False, **kwargs):
        sax_inputs = sphstack([bitmap.explode(self.inputs, i) for i in range(self.inputs.shape[1])])
        partitions = [{'contract_id': f'{i}', **p} for i, p in enumerate(self.partitions) for _ in range(bitmap.nf)]
        levels = np.ones(len(partitions))

        if inplace:
            self.update(inputs=sax_inputs, partitions=partitions, levels=levels, **kwargs)
        else:
            return self.copy(inputs=sax_inputs, partitions=partitions, levels=levels, **kwargs)

    def pop(self, ind):
        tmp = self[ind]
        l_idx = list(set(range(len(self))).difference({ind}))

        # Update data
        self.partitions = [self.partitions[i] for i in l_idx]
        self.inputs, self.levels = self.inputs[:, l_idx], self.levels[l_idx]

        return tmp

    def copy(self, **kwargs):
        return FgComponents(**{
            'inputs': self.inputs.copy(), 'partitions': copy(self.partitions), 'levels': self.levels.copy(), **kwargs
        })


@dataclass
class ConvexHullProba:
    counts: Optional[Dict[str, np.array]] = None

    def get_probas(self, comp: FgComponents, bitmap: BitMap):
        ax_support_bounds = bitmap.b2f(comp.inputs.astype(bool)).A

        if self.counts is None:
            ax_counts = np.ones(ax_support_bounds.shape)
        else:
            ax_counts = np.vstack([self.counts.get(d['id'], np.ones(bitmap.nf)) for d in comp.partitions])

        ax_p = ~ax_support_bounds / ax_counts

        return ax_p / ax_p.sum(axis=1)[:, np.newaxis]

    def add(self, comp: FgComponents, bitmap: BitMap):
        if self.counts is None:
            self.counts = {}

        ax_counts = bitmap.b2f(comp.inputs.astype(bool)).A
        self.counts = {
            d['id']: self.counts.get(d['id'], np.ones(bitmap.nf)) + ax_counts[i]
            for i, d in enumerate(comp.partitions)
        }
        return self


@dataclass
class DrainerFeedbacks:
    penalties: np.array
    rewards: np.array

    def get_all(self):
        return self.penalties, self.rewards


@dataclass
class DrainerParameters:
    total_size: int
    batch_size: int
    margin: float
    feedbacks: Optional[DrainerFeedbacks] = None
    weights: Optional[np.array] = None
    precisions: Optional[np.array] = None

    def limit_precisions(self):
        return (self.precisions - self.margin).clip(min=self.margin + 0.01)


@dataclass
class TrackerParameters:
    min_delta_area: float
    max_no_changes: int

