"""All classes that specify data structures."""
# Global import
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from numpy import array, empty, hstack
from scipy.sparse import spmatrix, csc_matrix, hstack as sphstack

# Local import

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
        l_idx = list(set(range(len(self))).difference({ind}))

        # Update data
        self.partitions = [self.partitions[i] for i in l_idx]
        self.inputs, self.levels = self.inputs[:, l_idx], self.levels[l_idx]

        return tmp


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

    def get_target_precisions(self):
        return (self.precisions - self.margin).clip(min=self.margin + 0.01)
