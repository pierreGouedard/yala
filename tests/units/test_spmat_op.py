# Global import
from scipy.sparse import csc_matrix
import numpy as np
import unittest

# Local import
from src.model.core.server import YalaUnclassifiedServer
from src.model.core.sampler import Sampler
from src.model.core.encoder import MultiEncoders
from src.model.utils.data_models import BitMap
from src.model.utils.data_models import FgComponents
from src.model.utils.spmat_op import shrink, expand, bounds, add_connex


class TestSpmatOp(unittest.TestCase):
    def setUp(self):
        # Create datasets
        self.origin_features = np.random.randn(20000, 2)
        self.basis = np.vstack([np.cos(np.arange(0, np.pi, 0.2)), np.sin(np.arange(0, np.pi, 0.2))])
        self.augmented_features = self.origin_features.dot(self.basis) * 100

        # Build model's element
        self.encoder = MultiEncoders(50, 'quantile', bin_missing=False)
        X_enc, y_enc = self.encoder.fit_transform(X=self.augmented_features, y=np.random.binomial(1, 0.5, 20000))
        self.server = YalaUnclassifiedServer(X_enc, y_enc).stream_features()
        self.bitmap = BitMap(self.encoder.bf_map, self.encoder.bf_map.shape[0], self.encoder.bf_map.shape[1])
        self.sampler = Sampler(self.server, self.bitmap)

        # Build test components
        self.test_components = self.build_random_comp()

        print("======= Test component input ======= ")
        print(self.bitmap.b2f(self.test_components.inputs).A)

    def build_random_comp(self):
        return self.sampler.init_sample(2, window_length=8)

    def test_expand(self):
        """
        python -m unittest tests.units.test_spmat_op.TestSpmatOp.test_expand

        """
        sax_expanded_inputs = expand(self.test_components.inputs, self.bitmap, n=3)

        # Test that features shouldn't have changed
        ax_expnd_features = self.bitmap.b2f(sax_expanded_inputs > 0).A.astype(bool)
        ax_test_features = self.bitmap.b2f(self.test_components.inputs > 0).A.astype(bool)
        self.assertTrue((ax_expnd_features == ax_test_features).all())

        # Test that correct number of bit has been added
        for i, sax_mask in enumerate(self.bitmap):
            for j in range(sax_expanded_inputs.shape[1]):
                n_expnd = sax_expanded_inputs[:, j].multiply(sax_mask).nnz
                n_test = self.test_components.inputs[:, j].multiply(sax_mask).nnz
                if n_test > 0:
                    self.assertTrue(n_expnd >= n_test + 3)
                    self.assertTrue(n_expnd <= n_test + 6)

    def test_shrink(self):
        """
        python -m unittest tests.units.test_spmat_op.TestSpmatOp.test_shrink

        """
        sax_shrinked_inputs = shrink(self.test_components.inputs, self.bitmap, n_shrink=2)

        # Test that features shouldn't have changed
        ax_shrinked_inputs = self.bitmap.b2f(sax_shrinked_inputs > 0).A.astype(bool)
        ax_test_features = self.bitmap.b2f(self.test_components.inputs > 0).A.astype(bool)
        self.assertTrue((ax_shrinked_inputs == ax_test_features).all())

        # Test that correct number of bit has been added
        for i, sax_mask in enumerate(self.bitmap):
            for j in range(sax_shrinked_inputs.shape[1]):
                n_shrinked = sax_shrinked_inputs[:, j].multiply(sax_mask).nnz
                n_test = self.test_components.inputs[:, j].multiply(sax_mask).nnz
                if n_test > 0:
                    self.assertTrue(n_shrinked == n_test - 4)

    def test_bounds(self):
        """
        python -m unittest tests.units.test_spmat_op.TestSpmatOp.test_bounds

        """
        sax_bounds_inputs = bounds(self.test_components.inputs, self.bitmap)

        # Test that features shouldn't have changed
        ax_bounds_inputs = self.bitmap.b2f(sax_bounds_inputs > 0).A.astype(bool)
        ax_test_features = self.bitmap.b2f(self.test_components.inputs > 0).A.astype(bool)
        self.assertTrue((ax_bounds_inputs == ax_test_features).all())

        # Test that correct number of bit has been added
        for i, sax_mask in enumerate(self.bitmap):
            for j in range(sax_bounds_inputs.shape[1]):
                if ax_test_features[j, i]:
                    self.assertEqual(sax_bounds_inputs[:, j].multiply(sax_mask).nnz, 2)

    def test_connex(self):
        """
        python -m unittest tests.units.test_spmat_op.TestSpmatOp.test_connex

        """
        pass

