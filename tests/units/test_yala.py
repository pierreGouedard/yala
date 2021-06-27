# Global import
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import numpy as np
from scipy.sparse import csc_matrix
import unittest

# Local import
from src.model.yala import Yala


class TestYala(unittest.TestCase):
    show_dataset = False
    p_yala = {
        'draining_margin': 0.1, 'n_node_by_iter': 2, 'level_0': 5, 'n_update': 2, 'draining_size': 200000,
        'batch_size': 100000, 'min_firing': 100, 'dropout_rate_mask': 0.99, 'max_iter': 100, 'n_bin': 10,
        'bin_method': 'quantile', 'bin_missing': False
    }

    def setUp(self):

        # Create datasets
        mult_norm = multivariate_normal(mean=np.zeros(2), cov=np.eye(2))
        self.origin_features = np.random.randn(20000, 2)

        # Compute labels
        self.target_circle_soft = np.array([np.random.binomial(1, mult_norm.pdf(x)) for x in self.origin_features])
        self.target_circle_hard = np.array([mult_norm.pdf(x) > 0.11 for x in self.origin_features])
        self.target_line_hard = np.array([x[1] < 0 for x in self.origin_features])

        # augment dataset
        # ax_basis = np.random.randn(2, 20) # random basis
        self.basis = np.vstack([np.cos(np.arange(0, np.pi, 0.2)), np.sin(np.arange(0, np.pi, 0.2))]) # circular basis
        self.augmented_features = self.origin_features.dot(self.basis) * 100

        # plot original dataset
        if self.show_dataset:
            self.plot_dataset(self.origin_features, self.target_circle_hard)

        # Instantiate visualizer
        self.perf_plotter = PerfPlotter(
            self.origin_features, self.target_circle_hard, list(range(len(self.target_circle_hard)))
        )

    @staticmethod
    def plot_dataset(ax_x, ax_y):
        plt.scatter(ax_x[ax_y > 0, 0], ax_x[ax_y > 0, 1], c='r', marker='+')
        plt.scatter(ax_x[ax_y == 0, 0], ax_x[ax_y == 0, 1], c='b', marker='o')
        plt.show()

    def test_circle_hard(self):
        """
        python -m unittest tests.units.test_yala.TestYala.test_circle_hard

        """
        model = Yala(**self.p_yala)
        model.fit(self.augmented_features, self.target_circle_hard, **{"perf_plotter": self.perf_plotter})


class PerfPlotter:

    def __init__(self, ax_x, ax_y, indices):
        self.x = ax_x
        self.y = ax_y
        self.indices = indices

    def __call__(self, ax_yhat):

        for i in range(ax_yhat.shape[1]):
            fig, (ax_got, ax_hat) = plt.subplots(1, 2)
            fig.suptitle(f'Viz GOT vs Preds #{i}')

            ax_got.scatter(self.x[self.y > 0, 0], self.x[self.y > 0, 1], c='r', marker='+')
            ax_got.scatter(self.x[self.y == 0, 0], self.x[self.y == 0, 1], c='b', marker='o')

            ax_hat.scatter(self.x[ax_yhat[:, i] > 0, 0], self.x[ax_yhat[:, i] > 0, 1], c='r', marker='+')
            ax_hat.scatter(self.x[ax_yhat[:, i] == 0, 0], self.x[ax_yhat[:, i] == 0, 1], c='b', marker='o')

            plt.show()



