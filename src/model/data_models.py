"""All classes that specify data structures."""
# Global import
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from numpy import array, empty, hstack
from scipy.sparse import spmatrix, csc_matrix, hstack as sphstack

# Local import


@dataclass
class AmplificationComponents:
    inputs: spmatrix
    partitions: List[Dict[str, Any]]
    bit_inner: spmatrix
    vertex_norm: array

    def __len__(self):
        return self.vertex_norm.shape[0]

    def __getitem__(self, i):
        assert isinstance(i, int), 'index should be an integer'
        return AmplificationComponents(
            inputs=self.inputs[:, i], partitions=[self.partitions[i]], bit_inner=self.bit_inner[[2 * i, 2 * i + 1], :],
            vertex_norm=self.vertex_norm[[i]]
        )

    def pop(self, ind):

        tmp = self[ind]

        # Get indices to keep
        l_idx = range(len(self))

        # Update data
        self.inputs = self.inputs[:, [i for i in l_idx if i != ind]]
        self.partitions = [self.partitions[i] for i in l_idx if i != ind]
        self.bit_inner = self.bit_inner[[i for i in range(2 * len(self)) if i not in {2 * ind, 2 * ind + 1}], :]
        self.vertex_norm = self.vertex_norm[[i for i in l_idx if i != ind]]

        return tmp


@dataclass
class FgComponents:
    inputs: spmatrix
    partitions: List[Dict[str, Any]]
    levels: array

    @staticmethod
    def empty_comp():
        return FgComponents(csc_matrix((0, 0)), [], empty((0,)))

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
        l_idx = set(range(len(self))).difference({ind})

        # Update data
        self.partitions = [self.partitions[i] for i in l_idx]
        self.inputs, self.levels = self.inputs[:, l_idx], self.levels[l_idx]

        return tmp


@dataclass
class DrainerFeedbacks:
    penalties: array
    rewards: array


@dataclass
class DrainerParameters:
    feedbacks: Optional[DrainerFeedbacks]
    weights: Optional[array]
