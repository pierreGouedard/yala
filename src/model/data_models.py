"""All classes that specify data structures."""
from dataclasses import dataclass
from typing import Optional
from numpy import array
from scipy.sparse import spmatrix


@dataclass
class TransientComponents:
    feature_precision: spmatrix
    feature_count: spmatrix
    inputs: spmatrix

    def __len__(self):
        return self.inputs.shape[1]


@dataclass
class BaseComponents:
    inputs: Optional[spmatrix]
    levels: Optional[array]
    precisions: array
    counts: array

    def __len__(self):
        return self.inputs.shape[1]

    def __iter__(self):
        for i in range(len(self)):
            yield self.precisions[i], self.counts[i]

    def reduce(self, idx):
        self.inputs, self.levels = self.inputs[:, idx], self.levels[idx]
        self.precisions, self.counts = self.precisions[idx], self.counts[idx]


@dataclass
class ExtractedDrainedComponents:
    base_components: Optional[BaseComponents]
    transient_components: Optional[TransientComponents]


@dataclass
class DrainerFeedbacks:
    penalties: array
    rewards: array


@dataclass
class DrainerParameters:
    feedbacks: Optional[DrainerFeedbacks]
    weights: Optional[array]
