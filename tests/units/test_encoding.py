# Global imports
import unittest
import numpy as np
import random

# Local import
from src.model.helpers.encoder import NumEncoder, CatEncoder


class TesEncoder(unittest.TestCase):

    def setUp(self):

        # Test variables for numerical encoder
        self.X_1 = np.array([
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [0, 1, 2, 10, 20, 30, 100, 101, 102, 103]
        ])

        self.n_bins_1, self.n_bins_2 = 5, 12
        self.n_quantiles = 3
        self.bounds = [1, 10]
        self.i = np.random.randint(0, self.X_1.shape[1])

        # Test variables for categorical encoder
        self.l_cats_str, self.l_cats_int, self.size = ['a', 'b', 'c'], [10, 0, 100], 10
        self.X_2 = np.array([
            [random.choice(self.l_cats_str) for _ in range(self.size)],
            [random.choice(self.l_cats_int) for _ in range(self.size)],
        ]).transpose()

    def test_encoder_numeric(self):
        """
        python -m unittest tests.units.test_encoding.TesEncoder.test_encoder_numeric

        """
        # Get signal for test encoder based on bounds definition
        X = self.X_1[[0], :]

        # Built discrete space and validate it
        num_encoder = NumEncoder(self.n_bins_1, method='bounds', bounds=self.bounds)
        num_encoder.fit(X)

        self.assertTrue(min(num_encoder.bins.values()) == min(self.bounds))
        self.assertTrue(max(num_encoder.bins.values()) == max(self.bounds))
        self.assertTrue(len(num_encoder.bins.values()) == self.n_bins_1)

        # Encode and validate
        sax_bits = num_encoder.transform(X)
        sax_bits_l = sax_bits[0, self.i * 2 * self.n_bins_1: ((self.i * 2) + 1) * self.n_bins_1]
        sax_bits_u = sax_bits[0, ((self.i * 2) + 1) * self.n_bins_1: (self.i + 1) * 2 * self.n_bins_1]

        self.assertEqual(sax_bits.shape[0], X.shape[0])
        self.assertEqual(sax_bits.shape[1], X.shape[1] * 2 * self.n_bins_1)
        self.assertTrue(all([sax_bits_l[0, i] for i, v in num_encoder.bins.items() if X[0, self.i] >= v]))
        self.assertTrue(all([sax_bits_u[0, i] for i, v in num_encoder.bins.items() if X[0, self.i] < v]))

        # Get signal for test encoder based on quantiles
        X = self.X_1[[1], :]

        # Built discrete space and validate it
        num_encoder = NumEncoder(self.n_bins_2, method='quantile', n_quantile=self.n_quantiles)
        num_encoder.fit(X)

        self.assertTrue(min(num_encoder.bins.values()) == X.min())
        self.assertTrue(max(num_encoder.bins.values()) == X.max())
        self.assertTrue(len(num_encoder.bins.values()) == self.n_bins_2)
        self.assertEqual(len([x for x in num_encoder.bins.values() if x < 10]), 5)
        self.assertEqual(len([x for x in num_encoder.bins.values() if x > 99]), 4)

    def test_encoder_category(self):
        """
        python -m unittest tests.units.test_encoding.TesEncoder.test_encoder_category

        """

        # Test fitting of encoder
        cat_encoder = CatEncoder()
        cat_encoder.fit(self.X_2)

        self.assertTrue(all([v in self.l_cats_str for v in cat_encoder.categories_[0]]))
        self.assertTrue(all([int(v) in self.l_cats_int for v in cat_encoder.categories_[1]]))

        # Test encoding
        sax_bits = cat_encoder.transform(self.X_2)
        self.assertEqual(sax_bits.shape[0], self.X_2.shape[0])
        self.assertEqual(sax_bits.shape[1], len(self.l_cats_str) + len(self.l_cats_int))

        # test inverse transform
        X = cat_encoder.inverse_transform(sax_bits)
        self.assertTrue((X == self.X_2).all())
