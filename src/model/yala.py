# Global import
from numpy import ones

# Local import
from src.model.core.firing_graph import YalaFiringGraph
from src.model.core.server import YalaUnclassifiedServer, YalaMisclassifiedServer
from src.model.core.data_models import DrainerParameters, TrackerParameters, FgComponents, ConvexHullProba, BitMap
from src.model.utils import init_sample
from src.model.core.encoder import MultiEncoders
from src.model.core.drainers.shaper import Shaper
from src.model.core.tracker import Tracker


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

        drainer = Shaper(
            self.server, self.bitmap, self.drainer_params, min_firing=self.min_firing,
            perf_plotter=kwargs.get('perf_plotter', None)
        )

        ## TODO: Things to do for this new version:
        #   * In step 1: CH as mask (full convex level) / shrinked base (level=1) drained:
        #       * Only bits of base bound between CH of mask & passed base are drained (=> CH computation needed)
        #       * The integrity & convexity of base should always be preserved after rule applied:
        #         * if a base bound is not support => remove bound from base
        #         * if none base bounds are support => keep the entire CH as base
        #         * if only (n-1) base is support => keep the (n-1) + 1 random (n=2 in 2D space, 3 in 3D, ...)
        #         * drained base & CH are outputed separately
        #   * In step 2: Remaining base (full convex level) is mask / CH (level=1) is drained:
        #       * Same rule of selection applies (except the rule on n-1 support)
        #       * support of drained CH & mask are merged to compose a new convex base.
        #   * Track convergence of nodes and iterates if any vertex that has note converged remains.

        n_ch_candidate = 2
        for i in range(self.n_run):
            print("[YALA]: Iteration {}".format(i))

            # Initial sampling
            base_components = init_sample(
                self.n_parallel, self.init_level, self.server, self.bitmap, window_length=7
            )
            ch_probas = ConvexHullProba()

            # Instantiate tracking
            tracker = Tracker(
                [d['id'] for d in base_components.partitions], tracker_params=self.tracker_params,
                n_features=self.encoder.bf_map.shape[1]
            )

            # Core loop
            i = 0
            while len(base_components) > 0 and i < self.max_iter:

                # Sample bounds for convex hull & update ch proba
                ch_components = YalaFiringGraph.from_fg_comp(base_components)\
                    .get_convex_hull(self.server, self.drainer_params.batch_size)

                ch_components = ch_components.sample(
                    ch_probas.get_probas(base_components, self.bitmap), n_ch_candidate,
                    {"levels": ones(len(ch_components)) * n_ch_candidate}
                )
                ch_probas.add(ch_components, self.bitmap.b2f(ch_components.inputs.astype(bool)))

                # Draining part 1: keep separated components
                base_components, ch_components = drainer.prepare(
                    base_components.shrink(self.bitmap, **{'levels': ones(len(base_components))}),
                    ch_components
                )\
                    .drain_all()\
                    .select(merge=False)
                drainer.reset()

                # Draining part 2: Merge components
                base_components = drainer.prepare(
                    ch_components.update(**{'levels': ones(len(ch_components))}),
                    base_components
                )\
                    .drain_all()\
                    .select(merge=True)
                drainer.reset()

                # Swap from compression to expansion & track metrics TODO: ?? rename da shit
                base_components = tracker.swap_components(base_components)

                if i % 5 == 0:
                    tracker.visualize_indicators()
                    if base_components is not None:
                        drainer.visualize_fg(YalaFiringGraph.from_fg_comp(base_components))
                    resp = input('would you like to activate visualisation now ?')

                    if resp == "yes":
                        drainer.plot_perf_enabled = True
                        drainer.advanced_plot_perf_enabled = True
                        drainer.plot_perf_enabled = True
                        drainer.advanced_plot_perf_enabled = True
                i += 1

                if base_components is None:
                    break

            tracker.visualize_indicators()
            complete_components = tracker.components
            for comp in complete_components:
                drainer.visualize_fg(YalaFiringGraph.from_fg_comp(comp))

        return self
