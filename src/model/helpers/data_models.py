"""All classes that specify data structures."""
# Global import
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from numpy import array, empty, hstack, ones
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

    def complement(self, sax_mask=None):
        sax_inputs = csc_matrix(self.inputs.A ^ ones(self.inputs.shape, dtype=bool))

        if sax_mask is not None:
            sax_inputs = sax_inputs.multiply(sax_mask)

        return FgComponents(inputs=sax_inputs, partitions=copy(self.partitions), levels=self.levels.copy())

    def copy(self):
        return FgComponents(inputs=self.inputs.copy(), partitions=copy(self.partitions), levels=self.levels.copy())


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
    min_precision_gain: float
    min_size_gain: float
    max_no_gain: int

