# Global import
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import numpy as np
import unittest

# Local import
from yala.yala import Yala
from tests.units.utils import PerfPlotter


# TODO: remove the notion of precision, it should always be one with a 5% tolerance
#       Maybe do smaller bs so that link are removed very fast (not sure about this one though)
#       create the sampler / target encoder
#       Modify encoder (nbin very closed too reality)

class TestHardShape(unittest.TestCase):
    show_dataset = True
    p_yala = {
        'draining_margin': 0.05, 'n_parallel': 1, 'n_bounds_start': 2, 'batch_size': 50000, 'n_run': 100, 'n_bin': 100,
        'bin_method': 'quantile', 'bin_missing': False
    }
    n_shape = 1
    type_basis = 'circle'
    shape = 'circle'

    def setUp(self):
        np.random.seed(12345)

        # Create datasets
        self.origin_features = np.random.uniform(low=[-2.5, -2.5], high=[2.5, 2.5], size=(50000, 2))
        self.setup_circle_shape()
        self.setup_square_shape()
        self.setup_triangle_shape()
        self.setup_bold_line_shape()

        # augment dataset
        if self.type_basis == 'random':
            self.basis = np.random.randn(2, 20)
        elif self.type_basis == 'circle':
            self.basis = np.vstack([np.cos(np.arange(0, np.pi, np.pi / 20)), np.sin(np.arange(0, np.pi, np.pi / 20))])

        self.augmented_features = self.origin_features.dot(self.basis) * 100

        # plot original dataset
        if self.show_dataset:
            self.plot_dataset(self.origin_features, self.ctargets, 'Circle shapes')
            self.plot_dataset(self.origin_features, self.stargets, 'Square shapes')
            self.plot_dataset(self.origin_features, self.ttargets, 'Triangle shapes')
            self.plot_dataset(self.origin_features, self.bltargets, 'Bold line shapes')

    def setup_circle_shape(self):
        # Create datasets
        l_norm = [multivariate_normal(mean=np.random.randn(2) * 0, cov=np.eye(2)) for _ in range(self.n_shape)]

        # Compute labels
        self.ctargets = np.array([max([m.pdf(x) for m in l_norm]) > 0.10 for x in self.origin_features])

    def setup_square_shape(self):
        l_centers = [(np.random.randn(2) * 0.5, 1.) for _ in range(self.n_shape)]

        def is_inside(x):
            return any([all([abs(x[i] - c[0][i]) < (c[1] / 2) for i in range(2)]) for c in l_centers])

        # Compute labels
        test = np.array([[np.cos(np.pi / 10), np.cos(6 * np.pi / 10)], [np.sin(np.pi / 10), np.sin(6 * np.pi / 10)]])
        self.stargets = np.array([is_inside(x) for x in self.origin_features.dot(test)])

    def setup_triangle_shape(self):
        center = (np.random.randn(2) * 0.5, 1.)
        ax_line = np.array([-1, -1])

        def is_inside(x):
            return all([x[i] - center[0][i] > 0 for i in range(2)]) and x.dot(ax_line) + 1 > 0

        # Compute labels
        test = np.array([[np.cos(np.pi / 10), np.cos(6 * np.pi / 10)], [np.sin(np.pi / 10), np.sin(6 * np.pi / 10)]])
        self.ttargets = np.array([is_inside(x) for x in self.origin_features.dot(test)])

    def setup_bold_line_shape(self):
        l_lines = [np.array([-1, -1]), np.array([1, 1])]

        def is_inside(x):
            return x.dot(l_lines[0]) + 1 > 0 and x.dot(l_lines[1]) - 0.5 > 0

        # Compute labels
        test = np.array([[np.cos(np.pi / 10), np.cos(6 * np.pi / 10)], [np.sin(np.pi / 10), np.sin(6 * np.pi / 10)]])
        self.bltargets = np.array([is_inside(x) for x in self.origin_features.dot(test)])

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

    def test_triangle_hard(self):
        """
        python -m unittest tests.units.test_hard_shape.TestHardShape.test_triangle_hard

        """
        # Instantiate visualizer
        self.perf_plotter = PerfPlotter(
            self.origin_features, self.ttargets, list(range(len(self.ttargets)))
        )

        model = Yala(**self.p_yala)
        model.fit(self.augmented_features, self.ttargets, **{"perf_plotter": self.perf_plotter})

    def test_bold_line_hard(self):
        """
        python -m unittest tests.units.test_hard_shape.TestHardShape.test_bold_line_hard

        """
        # Instantiate visualizer
        self.perf_plotter = PerfPlotter(
            self.origin_features, self.ttargets, list(range(len(self.bltargets)))
        )

        model = Yala(**self.p_yala)
        model.fit(self.augmented_features, self.bltargets, **{"perf_plotter": self.perf_plotter})

