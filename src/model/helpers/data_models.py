"""All classes that specify data structures."""
# Global import
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from numpy import array, empty, hstack, ones, maximum
from scipy.sparse import spmatrix, csc_matrix, hstack as sphstack
from copy import deepcopy as copy

# Local import


@dataclass
class FgComponents:
    inputs: spmatrix
    partitions: List[Dict[str, Any]]
    levels: array
    __idx: int = 0

    @staticmethod
    def empty_comp():
        return FgComponents(csc_matrix((0, 0)), [], empty((0,)))

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
            inputs=sax_inputs, partitions=self.partitions + other.partitions, levels=hstack([self.levels, other.levels])
        )

    def pop(self, ind):

        tmp = self[ind]

        # Get indices to keep
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
class ShaperProba:
    dim: Tuple[int, int]
    support_proba: float
    counts: Optional[array] = None
    probas: Optional[array] = None

    def set_probas(self, ax_support_mask):
        if self.counts is None:
            self.counts = ones(self.dim)

        ax_p = ax_support_mask * self.support_proba + (~ax_support_mask * 1. / self.counts)
        self.probas = ax_p / ax_p.sum(axis=0)

        return self

    def add(self, ax_support_mask: array):
        import IPython
        IPython.embed()
        if self.counts is None:
            self.counts = ones(self.dim) + ax_support_mask
        else:
            self.counts += ax_support_mask

        return self

    def remove(self, ax_support_mask: array):
        if self.counts is None:
            self.counts = ones(self.dim)
        else:
            self.counts -= ax_support_mask
        return self


@dataclass
class DrainerFeedbacks:
    penalties: array
    rewards: array

    def get_all(self):
        return self.penalties, self.rewards


@dataclass
class DrainerParameters:
    total_size: int
    batch_size: int
    margin: float
    feedbacks: Optional[DrainerFeedbacks] = None
    weights: Optional[array] = None
    precisions: Optional[array] = None

    def limit_precisions(self):
        return (self.precisions - self.margin).clip(min=self.margin + 0.01)


@dataclass
class TrackerParameters:
    min_delta_area: float
    max_no_changes: int

