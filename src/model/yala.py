# Global import

# Local import
from src.model.helpers.patterns import YalaBasePatterns
from src.model.helpers.server import YalaUnclassifiedServer, YalaMisclassifiedServer
from src.model.helpers.data_models import DrainerParameters, TrackerParameters, FgComponents
from src.model.utils import init_sample
from src.model.helpers.encoder import MultiEncoders
from src.model.helpers.operations.cleaner import Cleaner
from src.model.helpers.operations.shaper import Shaper
from src.model.helpers.tracker import Tracker


class Yala(object):

    def __init__(self,
                 n_run=5,
                 max_iter=20,
                 batch_size=1000,
                 draining_size=500,
                 draining_margin=0.05,
                 n_parallel=50,
                 init_level=1,
                 n_update=1,
                 dropout_rate_mask=0.2,
                 min_firing=10,
                 min_delta_area=0.05,
                 max_no_changes=3,
                 server_type='unclassified',
                 n_bin=10,
                 bin_method='quantile',
                 bin_missing=False,
    ):
        # Core parameters
        self.n_run = n_run
        self.max_iter = max_iter
        self.n_parallel = n_parallel
        self.init_level = init_level
        self.n_update = n_update
        self.min_firing = min_firing

        # Encoder params
        self.encoder = MultiEncoders(n_bin, bin_method, bin_missing=bin_missing)

        # Server params
        self.server_type, self.dropout_rate_mask, self.server = server_type, dropout_rate_mask, None

        # Drainer params
        self.drainer_params = DrainerParameters(total_size=draining_size, batch_size=batch_size, margin=draining_margin)

        # Tracker params
        self.tracker_params = TrackerParameters(min_delta_area=min_delta_area, max_no_changes=max_no_changes)

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

        # Instantiate core components
        if self.server_type == 'unclassified':
            self.server = YalaUnclassifiedServer(X_enc, y_enc[:, 1], dropout_rate_mask=self.dropout_rate_mask)\
                .stream_features()
        elif self.server_type == 'misclassified':
            self.server = YalaMisclassifiedServer(X_enc, y_enc[:, 1], dropout_rate_mask=self.dropout_rate_mask)\
                .stream_features()
        else:
            raise NotImplementedError

        shaper = Shaper(
            self.server, self.encoder.bf_map, self.drainer_params, min_firing=self.min_firing,
            perf_plotter=kwargs.get('perf_plotter', None)
        )

        cleaner = Cleaner(
            self.server, self.encoder.bf_map, self.drainer_params, min_firing=self.min_firing,
            perf_plotter=kwargs.get('perf_plotter', None)
        )

        for i in range(self.n_run):
            print("[YALA]: Iteration {}".format(i))

            # Initial sampling
            components = init_sample(
                self.n_parallel, self.init_level, self.server, self.encoder.bf_map, window_length=7
            )

            # Instantiate tracking
            tracker = Tracker(
                [d['id'] for d in components.partitions], tracker_params=self.tracker_params,
                n_features=self.encoder.bf_map.shape[1]
            )

            # Core loop
            i = 0
            while len(components) > 0 and i < self.max_iter:

                # Alter shape
                components = shaper.prepare(components).drain_all().select()
                shaper.reset()

                # Clean shape
                components = cleaner.prepare(components).drain_all().select()
                cleaner.reset()

                # Swap from compression to expansion & track metrics
                components = tracker.swap_components(components)

                if i % 5 == 0:
                    tracker.visualize_indicators()
                    if components is not None:
                        cleaner.visualize_fg(YalaBasePatterns.from_fg_comp(components))
                    resp = input('would you like to activate visualisation now ?')

                    if resp == "yes":
                        shaper.plot_perf_enabled = True
                        shaper.advanced_plot_perf_enabled = True
                        cleaner.plot_perf_enabled = True
                        cleaner.advanced_plot_perf_enabled = True
                i += 1

                if components is None:
                    break

            tracker.visualize_indicators()
            completes = tracker.components
            for comp in completes:
                cleaner.visualize_fg(YalaBasePatterns.from_fg_comp(comp))

        return self
