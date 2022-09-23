# Global import
import numpy as np
import unittest

# Local import
from firing_graph.servers import ArrayServer
from yala.encoder import MultiEncoders
from yala.utils.data_models import BitMap
from yala.sampler import Sampler
from tests.units.utils import PerfPlotter


class TestSampler(unittest.TestCase):

    def setUp(self):
        # Create datasets
        self.origin_features = np.random.randn(20000, 2)
        self.basis = np.vstack([np.cos(np.arange(0, np.pi, 0.2)), np.sin(np.arange(0, np.pi, 0.2))])
        self.augmented_features = self.origin_features.dot(self.basis) * 100
        self.targets = np.zeros(self.origin_features.shape[0])
        self.n_bounds = 2

        # Instantiate visualizer
        self.perf_plotter = PerfPlotter(
            self.origin_features, self.targets, list(range(len(self.targets)))
        )

        # Build model's element
        self.encoder = MultiEncoders(50, 'quantile', bin_missing=False)
        X_enc, y_enc = self.encoder.fit_transform(X=self.augmented_features, y=self.targets)
        self.server = ArrayServer(X_enc, y_enc).stream_features()
        self.bitmap = BitMap(self.encoder.bf_map, self.encoder.bf_map.shape[0], self.encoder.bf_map.shape[1])
        self.sampler = Sampler(self.server, self.bitmap, n_bounds=self.n_bounds)

    def test_init_sample(self):
        """
        python -m unittest tests.units.test_sampler.TestSampler.test_init_sample

        """
        comp_sampled = self.sampler.init_sample(3, n_bits=10)

        # Assert that number of bound is correct
        self.assertTrue((self.bitmap.b2f(comp_sampled.inputs.astype(bool)).A.sum(axis=1) == self.n_bounds).all())

        # Assert for each vertices / bounds there is at least 6 bit, at most 11 bit that are non null
        for i, comp in enumerate(comp_sampled):
            ax_cnt_features = self.bitmap.b2f(comp.inputs.astype(np.int32)).A[0, :]
            self.assertTrue((ax_cnt_features[ax_cnt_features > 0] >= 6).all())
            self.assertTrue((ax_cnt_features[ax_cnt_features > 0] <= 11).all())

    def test_sample_bounds(self):
        """
        python -m unittest tests.units.test_sampler.TestSampler.test_sample_bounds

        """
        # Init samples
        comp_sampled = self.sampler.init_sample(3, n_bits=10)

        # Sample additional bounds
        comp_test = self.sampler.sample_bounds(comp_sampled.copy(), 10000)

        # Assert that number of bound is correct
        ax_cnt_bounds_test = self.bitmap.b2f(comp_test.inputs.astype(bool)).A.sum(axis=1)
        ax_cnt_bounds_sampled = self.bitmap.b2f(comp_sampled.inputs.astype(bool)).A.sum(axis=1)
        self.assertTrue((ax_cnt_bounds_test == ax_cnt_bounds_sampled + self.n_bounds).all())

