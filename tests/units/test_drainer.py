# Global import
from scipy.sparse import csc_matrix
import numpy as np
import unittest

# Local import
from src.model.core.server import YalaUnclassifiedServer
from src.model.core.firing_graph import YalaFiringGraph
from src.model.core.encoder import MultiEncoders
from src.model.core.data_models import BitMap
from src.model.core.cleaner import Cleaner
from src.model.utils import init_sample


class TestCleaner(unittest.TestCase):

    def setUp(self):
        # Create datasets
        self.origin_features = np.random.randn(20000, 2)
        self.basis = np.vstack([np.cos(np.arange(0, np.pi, 0.2)), np.sin(np.arange(0, np.pi, 0.2))])
        self.augmented_features = self.origin_features.dot(self.basis) * 100
        self.targets = np.zeros(self.origin_features.shape[0])

        # Build model's element
        self.encoder = MultiEncoders(50, 'quantile', bin_missing=False)
        X_enc, y_enc = self.encoder.fit_transform(X=self.augmented_features, y=self.targets)
        self.server = YalaUnclassifiedServer(X_enc, y_enc).stream_features()
        self.bitmap = BitMap(self.encoder.bf_map, self.encoder.bf_map.shape[0], self.encoder.bf_map.shape[1])
        self.cleaner = Cleaner(self.server, self.bitmap, 3000)

        # Build test components
        self.test_component = init_sample(1, 2, self.server, self.bitmap, window_length=10)
        self.test_component_ch = YalaFiringGraph.from_fg_comp(self.test_component)\
            .get_convex_hull(self.server, 3000)

        print("======= Test component input ======= ")
        print(self.bitmap.b2f(self.test_component.inputs).A)
        print("======= Test component CH input ======= ")
        print(self.bitmap.b2f(self.test_component_ch.inputs).A)

    def sample(self, n):
        # Select 1 additional bound to clean
        ind_mask = list(np.random.choice(np.arange(self.bitmap.nf), size=n, replace=False))
        sax_inputs = csc_matrix(self.bitmap.bf_map[:, ind_mask].sum(axis=1)).multiply(self.test_component_ch.inputs)
        return sax_inputs + self.test_component.inputs[:, 0]

    def test_one_to_clean(self):
        """
        python -m unittest tests.units.test_cleaner.TestCleaner.test_one_to_clean

        """
        # Select 1 additional bound to clean
        sax_inputs = self.sample(1)
        print("======= CH sampled ======= ")
        print(self.bitmap.b2f(sax_inputs).A)

        # Clean component
        clean_component = self.cleaner.clean_component(
            self.test_component_ch.copy(inputs=sax_inputs, levels=np.ones(1) * 3)
        )

        self.assertTrue(
            (self.bitmap.b2f(self.test_component.inputs > 0).A == self.bitmap.b2f(clean_component.inputs > 0).A).all()
        )
        self.assertTrue((clean_component.levels == self.test_component.levels).all())

    def test_two_to_clean(self):
        """
        python -m unittest tests.units.test_cleaner.TestCleaner.test_two_to_clean

        """
        # Select 2 additional bound to clean
        sax_inputs = self.sample(2)
        print("======= CH sampled ======= ")
        print(self.bitmap.b2f(sax_inputs).A)

        # Clean component
        clean_component = self.cleaner.clean_component(
            self.test_component_ch.copy(inputs=sax_inputs, levels=np.ones(1) * 4)
        )

        self.assertTrue(
            (self.bitmap.b2f(self.test_component.inputs > 0).A == self.bitmap.b2f(clean_component.inputs > 0).A).all()
        )
        self.assertTrue((clean_component.levels == self.test_component.levels).all())

    def test_random_to_clean(self):
        """
        python -m unittest tests.units.test_cleaner.TestCleaner.test_random_to_clean

        """
        # Select n random additional bound to clean
        n = np.random.randint(3, 8, 1)
        sax_inputs = self.sample(n)
        print("======= CH sampled ======= ")
        print(self.bitmap.b2f(sax_inputs).A)

        # Clean component
        clean_component = self.cleaner.clean_component(
            self.test_component_ch.copy(inputs=sax_inputs, levels=np.ones(1) * 4)
        )
