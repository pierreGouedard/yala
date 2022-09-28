# Global import

# Local import
from yala.utils.data_models import DrainerParameters, TrackerParameters
from .encoder import MultiEncoders
from .drainers.shaper import Shaper
from .cleaner import Cleaner
from .tracker import Tracker
from .server import YalaServer
from .utils.visual import Visualizer


class Yala(object):

    def __init__(self,
                 n_run=5,
                 batch_size=1000,
                 draining_margin=0.05,
                 n_parallel=50,
                 n_bounds_start=6,
                 n_bounds_incr=2,
                 min_diff_area=0.05,
                 max_inactive=2,
                 n_bin=10,
                 bin_method='quantile',
                 bin_missing=False,
                 ):
        # Core parameters
        self.n_run = n_run
        self.n_parallel = n_parallel
        self.n_bounds_start, self.n_bounds_incr = n_bounds_start, n_bounds_incr

        # Encoder params
        self.encoder = MultiEncoders(n_bin, bin_method, bin_missing=bin_missing)
        self.server = None

        # Drainer params
        self.drainer_params = DrainerParameters(batch_size=batch_size, margin=draining_margin)

        # Tracker params
        self.tracker_params = TrackerParameters(min_diff_area=min_diff_area, max_inactive=max_inactive, min_area=5)

        # Yala Core attributes
        self.firing_graph = None

    def fit(self, X, y, **kwargs):
        """
        row

        :param X:
        :param y:
        :return:
        """
        # encode: X is a numpy/scipy array, y in numpy/scipy array
        X_enc, y_enc = self.encoder.fit_transform(X=X, y=y)
        self.server = YalaServer(X_enc, y_enc, self.encoder.bf_map, self.n_bounds_start, self.n_bounds_incr)

        # Gte core components
        visualizer = Visualizer(kwargs.get('perf_plotter', None))
        shaper = Shaper(self.server, self.drainer_params, False, visualizer)
        cleaner = Cleaner(self.server, self.drainer_params.batch_size)

        for i in range(self.n_run):
            print("[YALA]: Iteration {}".format(i))

            # Initial sampling
            base_components = self.server.init_sampling(self.n_parallel, n_bits=7)

            # Instantiate tracking
            tracker = Tracker(self.tracker_params, self.server).update_metrics(base_components, update_tracking=True)

            # Core loop
            stop = False
            while not stop:
                # Shape components
                base_components = shaper.shape(base_components)

                # Clean base components
                base_components = cleaner.clean_component(base_components)

                # Check for component that has converged
                base_components = tracker.refresh_tracking(base_components)

                # Exit loop if comps is empty or all bounds explored otherwise sample new bounds
                stop = self.server.update_bounds(base_components)

            # Visualize kpis
            tracker.visualize_indicators()
            complete_components = tracker.conv_comps
            if not base_components.empty:
                complete_components += base_components

            visualizer.visualize_comp(complete_components, self.server)

        return self
