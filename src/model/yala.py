# Global import

# Local import
from src.model.helpers.patterns import YalaBasePatterns
from src.model.helpers.server import YalaUnclassifiedServer, YalaMisclassifiedServer
from src.model.helpers.data_models import DrainerParameters, TrackerParameters, FgComponents
from src.model.utils import init_sample
from src.model.helpers.encoder import MultiEncoders
from src.model.helpers.operations.refiner import Refiner
from src.model.helpers.operations.expander import Expander
from src.model.helpers.tracker import Tracker


class Yala(object):

    def __init__(self,
                 max_iter=5,
                 batch_size=1000,
                 draining_size=500,
                 draining_margin=0.05,
                 n_parallel=50,
                 init_level=1,
                 n_update=1,
                 dropout_rate_mask=0.2,
                 min_firing=10,
                 min_precision_gain=0.01,
                 min_size_gain=0.05,
                 max_no_gain=2,
                 server_type='unclassified',
                 n_bin=10,
                 bin_method='quantile',
                 bin_missing=False,
    ):

        # Core parameters
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
        self.tracker_params = TrackerParameters(
            min_precision_gain=min_precision_gain, min_size_gain=min_size_gain, max_no_gain=max_no_gain
        )

        # Yala Core attributes
        self.firing_graph = None

        # TODO: test / temp
        self.d_merger = {}

    def fit(self, X, y, **kwargs):
        """
        row

        :param X:
        :param y:
        :param eval_set:
        :param sample_weight:
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

        # TODO: Given the new server, each time a global mask is applied, we should update the draining and batch size
        # TODO: Maintain a pool of candidate features (updated with refiner and expander)

        refiner = Refiner(
            self.server, self.encoder.bf_map, self.drainer_params, min_firing=self.min_firing, n_update=self.n_update,
            perf_plotter=kwargs.get('perf_plotter', None), plot_perf_enabled=True
        )

        expander = Expander(
            self.server, self.encoder.bf_map, self.drainer_params, min_firing=self.min_firing,
            perf_plotter=kwargs.get('perf_plotter', None), plot_perf_enabled=True
        )

        for i in range(self.max_iter):
            print("[YALA]: Iteration {}".format(i))

            # Initial sampling
            component = init_sample(
                self.n_parallel, self.init_level, self.server, self.encoder.bf_map, window_length=3
            )

            # Instantiate tracking
            tracker = Tracker(
                [d['id'] for d in component.partitions], min_firing=self.min_firing, tracker_params=self.tracker_params
            )

            # Core loop
            i = 0
            while len(component) > 0:

                # Refine
                component = refiner.prepare(component).drain_all().select()
                refiner.reset()

                # Expand
                component = expander.prepare(component).drain_all().select()
                expander.reset()

                # Track metrics
                component = tracker.pop_complete(component)
                i += 1
                print(i)

                if component is None:
                    print('here')
                    import IPython
                    IPython.embed()

                if i % 5 == 0:
                    print('there')
                    tracker.visualize_indicators()
                    import IPython
                    IPython.embed()

            import IPython
            IPython.embed()
            # Augment current firing graph
            completes = tracker.get_complete_component()
            if self.firing_graph is not None:
                self.firing_graph = self.firing_graph.augment_from_fg_comp(completes or FgComponents.empty_comp())

            else:
                self.firing_graph = YalaBasePatterns.from_fg_comp(completes or FgComponents.empty_comp())

            # Update server
            if len(completes) > 0:
                self.server.update_mask_with_pattern(YalaBasePatterns.from_fg_comp(completes).reduce_output())
            self.server.sax_mask_forward = None
            self.server.pattern_backward = None

        import IPython
        IPython.embed()

        return self

    def predict_proba(self, X):

        assert self.firing_graph is not None, "First fit firing graph"

        sax_x = self.firing_graph.propagate(X)
        ax_probas = self.d_merger['C=0.1,fit_intercept=True,penalty=l2'].predict_proba(sax_x.A)
        return ax_probas[:, [1]]

    def predict(self, X, n_label, min_probas=0.1):
        pass