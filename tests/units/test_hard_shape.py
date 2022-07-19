# Global import
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import numpy as np
import unittest

# Local import
from src.model.yala import Yala
from tests.units.utils import PerfPlotter


class TestHardShape(unittest.TestCase):
    show_dataset = True
    p_yala = {
        'draining_margin': 0.05, 'n_parallel': 1, 'init_level': 3, 'n_update': 2, 'draining_size': 10000,
        'batch_size': 5000, 'min_firing': 100, 'dropout_rate_mask': 0.99, 'n_run': 100, 'n_bin': 50,
        'bin_method': 'quantile', 'bin_missing': False
    }
    n_shape = 1
    type_basis = 'circle'
    shape = 'circle'

    def setUp(self):
        np.random.seed(1234)

        # Create datasets
        self.origin_features = np.random.randn(20000, 2)
        self.setup_circle_shape()
        self.setup_square_shape()

        # augment dataset
        if self.type_basis == 'random':
            self.basis = np.random.randn(2, 20)
        elif self.type_basis == 'circle':
            self.basis = np.vstack([np.cos(np.arange(0, np.pi, 0.2)), np.sin(np.arange(0, np.pi, 0.2))])

        self.augmented_features = self.origin_features.dot(self.basis) * 100

        # plot original dataset
        if self.show_dataset:
            self.plot_dataset(self.origin_features, self.ctargets, 'Circle shapes')
            self.plot_dataset(self.origin_features, self.stargets, 'Square shapes')

    def setup_circle_shape(self):
        # Create datasets
        l_norm = [multivariate_normal(mean=np.random.randn(2) * 0, cov=np.eye(2)) for _ in range(self.n_shape)]

        # Compute labels
        self.ctargets = np.array([max([m.pdf(x) for m in l_norm]) > 0.10 for x in self.origin_features])

    def setup_square_shape(self):
        l_centers = [(np.random.randn(2), abs(np.random.randn())) for _ in range(self.n_shape)]

        def is_inside(x):
            return any([all([abs(x[i] - c[0][i]) < (c[1] / 2) for i in range(2)]) for c in l_centers])

        # Compute labels
        self.stargets = np.array([is_inside(x) for x in self.origin_features])

    @staticmethod
    def plot_dataset(ax_x, ax_y, title):
        plt.scatter(ax_x[ax_y > 0, 0], ax_x[ax_y > 0, 1], c='r', marker='+')
        plt.scatter(ax_x[ax_y == 0, 0], ax_x[ax_y == 0, 1], c='b', marker='o')
        plt.title(title)
        plt.show()

    def test_circle_hard(self):
        """
        python -m unittest tests.units.test_hard_shape.TestHardShape.test_circle_hard

        """
        # Instantiate visualizer
        self.perf_plotter = PerfPlotter(
            self.origin_features, self.ctargets, list(range(len(self.ctargets)))
        )

        model = Yala(**self.p_yala)
        model.fit(self.augmented_features, self.ctargets, **{"perf_plotter": self.perf_plotter})

    def test_square_hard(self):
        """
        python -m unittest tests.units.test_hard_shape.TestHardShape.test_square_hard

        """
        # Instantiate visualizer
        self.perf_plotter = PerfPlotter(
            self.origin_features, self.stargets, list(range(len(self.stargets)))
        )

        model = Yala(**self.p_yala)
        model.fit(self.augmented_features, self.stargets, **{"perf_plotter": self.perf_plotter})
