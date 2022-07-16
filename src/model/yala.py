# Global import
from numpy import ones

# Local import
from src.model.core.server import YalaUnclassifiedServer, YalaMisclassifiedServer
from src.model.utils.data_models import DrainerParameters, TrackerParameters, BitMap
from src.model.core.encoder import MultiEncoders
from src.model.core.drainers.shaper import Shaper
from src.model.core.cleaner import Cleaner
from src.model.core.tracker import Tracker
from src.model.core.sampler import Sampler


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

        # Instantiate core components
        if self.server_type == 'unclassified':
            self.server = YalaUnclassifiedServer(X_enc, y_enc, dropout_rate_mask=self.dropout_rate_mask)\
                .stream_features()
        elif self.server_type == 'misclassified':
            self.server = YalaMisclassifiedServer(X_enc, y_enc, dropout_rate_mask=self.dropout_rate_mask)\
                .stream_features()
        else:
            raise NotImplementedError

        ## TODO: Again here we are going through a fundamental refactor of the procedure:
        ##  in this new procedure we will:
        #       1- "move" 1 bound at a time DONE
        #       2- optimize the cleaner so no draining is necessary + remove 1 bound at a time DONE
        #       3- Make sure the Integrity of convex component is preserved at all time TODO
        #
        #   Task 1:
        #       Refactor the cleaner - DONE
        #   Task 2:
        #       Refactor the Yala drainer so it handle the logic of 1 at a time => TOTEST
        #   Task 3:
        #       Refactor Yala drainer so it handle the preservation of integrity of convex component => TODO
        #   Task 4:
        #       Change & adapt the core of YALA (this file, basically) => TOTEST

        shaper = Shaper(
            self.server, self.bitmap, self.drainer_params, min_firing=self.min_firing,
            perf_plotter=kwargs.get('perf_plotter', None), plot_perf_enabled=True,
            advanced_plot_perf_enabled=True
        )
        cleaner = Cleaner(self.server, self.bitmap, self.drainer_params.batch_size)
        sampler = Sampler(self.server, self.bitmap)

        n_ch_candidate = 2
        for i in range(self.n_run):
            print("[YALA]: Iteration {}".format(i))

            # Initial sampling
            base_components = sampler.init_sample(self.n_parallel, window_length=7)

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

                # Resample new bounds for base components that have not converged yet
                sampler.sample_bounds(base_components, self.drainer_params.batch_size)

                if i % 5 == 0:
                    tracker.visualize_indicators()
                    if base_components is not None:
                        shaper.visualize_comp(base_components)
                    resp = input('would you like to activate visualisation now ?')
                    if resp == "yes":
                        shaper.plot_perf_enabled = True
                        shaper.advanced_plot_perf_enabled = True

                i += 1

                if base_components is None:
                    break

            tracker.visualize_indicators()
            complete_components = tracker.components
            shaper.visualize_comp(complete_components)

        return self
