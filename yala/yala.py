# Global import

# Local import
from firing_graph.servers import ArrayServer
from yala.utils.data_models import DrainerParameters, TrackerParameters, BitMap
from .encoder import MultiEncoders
from .drainers.shaper import Shaper
from .cleaner import Cleaner
from .tracker import Tracker
from .sampler import Sampler


class Yala(object):

    def __init__(self,
                 n_run=5,
                 max_iter=20,
                 batch_size=1000,
                 draining_size=500,
                 draining_margin=0.05,
                 n_parallel=50,
                 n_update=1,
                 min_delta_area=0.05,
                 max_no_changes=3,
                 n_bin=10,
                 bin_method='quantile',
                 bin_missing=False,
                 ):
        # Core parameters
        self.n_run = n_run
        self.max_iter = max_iter
        self.n_parallel = n_parallel
        self.n_update = n_update

        # Encoder params
        self.encoder = MultiEncoders(n_bin, bin_method, bin_missing=bin_missing)
        self.server = None

        # Drainer params
        self.drainer_params = DrainerParameters(total_size=draining_size, batch_size=batch_size, margin=draining_margin)

        # Tracker params
        self.tracker_params = TrackerParameters(min_delta_area=min_delta_area, max_no_changes=max_no_changes)

        # Yala Core attributes
        self.firing_graph, self.bitmap = None, None

    def fit(self, X, y, **kwargs):
        """
        row

        :param X:
        :param y:
        :return:
        """
        # encode: X is a numpy/scipy array, y in numpy/scipy array
        X_enc, y_enc = self.encoder.fit_transform(X=X, y=y)
        self.bitmap = BitMap(self.encoder.bf_map, self.encoder.bf_map.shape[0], self.encoder.bf_map.shape[1])
        self.server = ArrayServer(X_enc, y_enc).stream_features()

        shaper = Shaper(
            self.server, self.bitmap, self.drainer_params, perf_plotter=kwargs.get('perf_plotter', None),
            plot_perf_enabled=True, advanced_plot_perf_enabled=False
        )
        cleaner = Cleaner(self.server, self.bitmap, self.drainer_params.batch_size)
        sampler = Sampler(self.server, self.bitmap)

        for i in range(self.n_run):
            print("[YALA]: Iteration {}".format(i))

            # Initial sampling
            base_components = sampler.init_sample(self.n_parallel, n_bits=7)

            # Instantiate tracking
            tracker = Tracker(
                [d['id'] for d in base_components.partitions], tracker_params=self.tracker_params,
                n_features=self.encoder.bf_map.shape[1]
            )

            # Core loop
            i = 0
            while len(base_components) > 0 and i < self.max_iter:
                # Shape components
                base_components = shaper.shape(base_components)
                shaper.reset()

                # Clean base components
                base_components = cleaner.clean_component(base_components)

                # Check for component that has converged
                base_components = tracker.swap_components(base_components)

                # exit loop if all comp have merged
                if base_components.empty:
                    break

                # Resample new bounds for base components that have not converged yet
                sampler.sample_bounds(base_components, self.drainer_params.batch_size)

                # Visualize training
                i += 1
                if i % 5 == 0:
                    tracker.visualize_indicators()

            tracker.visualize_indicators()
            complete_components = tracker.components
            shaper.visualize_comp(complete_components)

        return self
