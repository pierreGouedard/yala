# Global imports
import numpy as np
from scipy.sparse import csc_matrix

# Local import
from firing_graph.tools.helpers.sampler import Sampler


class YalaSampler(Sampler):
    def __init__(self, sax_map_fi, n_label, p_sample=0.8, verbose=0):
        # Get specific attribute
        self.map_fi = sax_map_fi
        (n_inputs, self.n_features) = self.map_fi.shape
        self.sax_inputs = None

        # Init parent class
        super(YalaSampler, self).__init__(n_label=n_label, n_inputs=n_inputs, p_sample=p_sample, verbose=verbose)

    def sample(self, patterns=None):
        # Initialize variables
        n, sax_input_mask = self.n_label if patterns is None else patterns.I.shape[1], None

        if patterns is not None:
            sax_feature_mask = self.map_fi.transpose().dot(patterns.I)
            sax_input_mask = (self.map_fi.dot(sax_feature_mask).astype(int) - patterns.I) > 0

        # Sampled feature and derive inputs
        sax_feature_mask = csc_matrix(np.random.binomial(1, self.p_sample, (self.n_features, n)).astype(bool))
        self.sax_inputs = self.map_fi.dot(sax_feature_mask)

        # mask inputs with patterns inputs
        if sax_input_mask is not None:
            self.sax_inputs = (self.sax_inputs.astype(int) - sax_input_mask) > 0

        print("[Sampler]: {} sampling of features.".format(n))

        return self
