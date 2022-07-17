# Global import
from scipy.sparse import csc_matrix
import numpy as np
import unittest
import matplotlib.pyplot as plt

# Local import
from src.model.core.server import YalaUnclassifiedServer
from src.model.core.drainers.shaper import Shaper
from src.model.core.encoder import MultiEncoders
from src.model.utils.data_models import BitMap, DrainerParameters
from src.model.utils.data_models import FgComponents
from src.model.utils.linalg import shrink


class TestDrainer(unittest.TestCase):
    width = 1
    plot_targets = True

    def setUp(self):
        # Create datasets
        self.origin_features = np.random.randn(20000, 2)
        self.basis = np.vstack([np.cos(np.arange(0, np.pi, 0.2)), np.sin(np.arange(0, np.pi, 0.2))])
        self.augmented_features = self.origin_features.dot(self.basis) * 100
        self.setup_square_shape()

        # Build model's element
        self.encoder = MultiEncoders(50, 'quantile', bin_missing=False)
        X_enc, y_enc = self.encoder.fit_transform(X=self.augmented_features, y=self.targets)
        self.server = YalaUnclassifiedServer(X_enc, y_enc).stream_features()
        self.bitmap = BitMap(self.encoder.bf_map, self.encoder.bf_map.shape[0], self.encoder.bf_map.shape[1])

        if self.plot_targets:
            self.plot_dataset(self.origin_features, self.targets, 'Circle shapes')

        # Build test components
        self.got_components = self.build_got_comp()
        self.shrink_components = self.build_shrink_comp()

        # Instantiate shaper
        self.perf_plotter = PerfPlotter(
            self.origin_features, self.targets, list(range(len(self.targets)))
        )
        self.drainer_params = DrainerParameters(total_size=20000, batch_size=10000, margin=0.05)
        self.shaper = Shaper(
            self.server, self.bitmap, self.drainer_params,  min_firing=10, perf_plotter=self.perf_plotter,
            plot_perf_enabled=True, advanced_plot_perf_enabled=True
        )

        if self.plot_targets:
            self.shaper.visualize_comp(self.got_components)
            self.shaper.visualize_comp(self.shrink_components)

        print("======= GOT component input ======= ")
        print(self.bitmap.b2f(self.got_components.inputs).A)
        print("======= Shrink component input ======= ")
        print(self.bitmap.b2f(self.shrink_components.inputs).A)

    def setup_square_shape(self):
        def is_inside(x):
            return all([abs(x[i]) < (self.width / 2) for i in range(2)])
        # Compute labels
        self.targets = np.array([is_inside(x) for x in self.origin_features])

    @staticmethod
    def plot_dataset(ax_x, ax_y, title):
        plt.scatter(ax_x[ax_y > 0, 0], ax_x[ax_y > 0, 1], c='r', marker='+')
        plt.scatter(ax_x[ax_y == 0, 0], ax_x[ax_y == 0, 1], c='b', marker='o')
        plt.title(title)
        plt.show()

    def build_got_comp(self):
        sax_x = self.server.next_forward(n=20000, update_step=False).sax_data_forward
        sax_y = self.server.next_backward(n=20000, update_step=False).sax_data_backward
        sax_inputs = sax_x.T.dot(sax_y[:, 1]).multiply(csc_matrix(self.bitmap.bf_map[:, [0, 8]].sum(axis=1)))
        return FgComponents(inputs=(sax_inputs > 0).astype(int), levels=np.array([2]), partitions=[{'id': 'got'}])

    def build_shrink_comp(self):
        return self.got_components.copy(
            inputs=shrink(self.got_components.inputs, self.bitmap, p_shrink=0.5).astype(int),
            partitions=[{"id": "shrink"}]
        )

    def test_draining(self):
        """
        python -m unittest tests.units.test_drainer.TestDrainer.test_draining

        """
        shaped_components = self.shaper.shape(self.shrink_components)
        self.shaper.reset()
        # TODO: Visual inspection + => assertion of the final area of shaped comp.
        import IPython
        IPython.embed()


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