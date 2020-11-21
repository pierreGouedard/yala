"""All classes that specify data structures."""
from dataclasses import dataclass
from typing import List, Dict, Optional
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
    precisions: Optional[array]
    counts: Optional[array] = None

    def __len__(self):
        return self.inputs.shape[1]

    def reduce(self, idx):
        self.inputs, self.levels, self.precisions = self.inputs[:, idx], self.levels[idx], self.precisions[idx]


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
