# Global import
import numpy as np
import unittest
import matplotlib.pyplot as plt

# Local import
from firing_graph.servers import ArrayServer
from yala.drainers.shaper import Shaper
from yala.encoder import MultiEncoders
from yala.utils.data_models import BitMap, DrainerParameters
from yala.utils.data_models import FgComponents
from yala.linalg.spmat_op import shrink
from tests.units.utils import PerfPlotter


class TestShaper(unittest.TestCase):
    width = 1
    plot_targets = True
    advanced_shaper_plot = False
    shaper_plot = True

    def setUp(self):
        # Create datasets
        self.origin_features = np.random.randn(20000, 2)
        self.basis = np.vstack([np.cos(np.arange(0, np.pi, 0.2)), np.sin(np.arange(0, np.pi, 0.2))])
        self.augmented_features = self.origin_features.dot(self.basis) * 100
        self.setup_square_shape()

        # Build model's element
        self.encoder = MultiEncoders(50, 'quantile', bin_missing=False)
        X_enc, y_enc = self.encoder.fit_transform(X=self.augmented_features, y=self.targets)
        self.server = ArrayServer(X_enc, y_enc).stream_features()
        self.bitmap = BitMap(self.encoder.bf_map, self.encoder.bf_map.shape[0], self.encoder.bf_map.shape[1])

        if self.plot_targets:
            self.plot_dataset(self.origin_features, self.targets, 'Square shapes')

        # Build test components
        self.got_components = self.build_got_comp()
        self.shrink_components = self.build_shrink_comp()

        # Instantiate shaper
        self.perf_plotter = PerfPlotter(
            self.origin_features, self.targets, list(range(len(self.targets)))
        )
        self.drainer_params = DrainerParameters(total_size=20000, batch_size=10000, margin=0.05)
        self.shaper = Shaper(
            self.server, self.bitmap, self.drainer_params,  perf_plotter=self.perf_plotter,
            plot_perf_enabled=self.shaper_plot, advanced_plot_perf_enabled=self.advanced_shaper_plot
        )

        if self.plot_targets:
            self.shaper.visualize_comp(self.got_components)
            self.shaper.visualize_comp(self.shrink_components)

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
        sax_inputs = sax_x.T.dot(sax_y[:, 1]).multiply(self.bitmap.bf_map[:, 0] + self.bitmap.bf_map[:, 8])
        return FgComponents(
            inputs=sax_inputs.tocsr(), mask_inputs=sax_inputs.tocsr(),
            levels=np.array([2]), partitions=[{'id': 'got'}]
        )

    def build_shrink_comp(self):
        shrink_inputs = shrink(self.got_components.inputs, self.bitmap, n_shrink=5)
        return self.got_components.copy(inputs=shrink_inputs, mask_inputs=shrink_inputs, partitions=[{"id": "shrink"}])

    def test_shaping(self):
        """
        python -m unittest tests.units.test_shaper.TestShaper.test_shaping

        """
        # Shape components
        shaped_components = self.shaper.shape(self.shrink_components)

        # Validate shaping
        got_area = self.got_components.inputs.sum(axis=0).A[0, :] / \
            (self.bitmap.b2f(self.got_components.inputs.astype(bool)).A.sum(axis=1) + 1e-6)

        self.assertTrue(abs(shaped_components.partitions[0]['area'] - got_area[0]) < 2.5)
        self.assertAlmostEqual(shaped_components.partitions[0]['precision'], 1, delta=0.1)
