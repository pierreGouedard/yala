# Global import
from firing_graph.solver.drainer import FiringGraphDrainer
import numpy as np
from scipy.sparse import lil_matrix, csc_matrix
from dataclasses import asdict
import matplotlib.pyplot as plt

# Local import
from src.model.patterns import YalaBasePatterns
from src.model.helpers.server import YalaUnclassifiedServer, YalaMisclassifiedServer
from .data_models import DrainerFeedbacks, DrainerParameters
from src.model.helpers.amplifiers.drain import DrainAmplifier
from src.model.helpers.amplifiers.refine import RefineAmplifier
from src.model.utils import init_sample, prepare_refinement, prepare_expansion
from src.model.data_models import FgComponents
from src.model.helpers.encoder import MultiEncoders


# TODO: Schedule of prod implementation of new paradigm:
#   * 3. implement final selection of vertices:
#       3.A. Remove duplicated factors ?
#       3.B. Create reducer (mean, regul mean, logistic reg)
#       3.C. Create encoding encoder
#       3.D. Make it up to modern standard (MYPY, flake8, pytest)

# TODO: finally clean the module and make it possible to use datalab to test data bases here


class Yala(object):

    def __init__(self,
                 max_iter=5,
                 batch_size=1000,
                 draining_size=500,
                 draining_margin=0.1,
                 n_node_by_iter=50,
                 level_0=1,
                 n_update=2,
                 dropout_rate_mask=0.2,
                 min_firing=10,
                 server_type='misclassified',
                 n_bin=10,
                 bin_method='quantile',
                 bin_missing=False,
                 n_augment=0,
                 basis='norm'
    ):

        # Core parameters
        self.max_iter = max_iter
        self.level_0 = level_0
        self.n_update = n_update
        self.n_node_by_iter = n_node_by_iter
        self.min_firing = min_firing

        # Encoder params
        self.encoder = MultiEncoders(n_bin, bin_method, bin_missing=bin_missing, n_augment=n_augment, basis=basis)

        # Server params
        self.server_type = server_type
        self.dropout_rate_mask = dropout_rate_mask

        # Drainer params
        self.drainer_params = DrainerParameters(feedbacks=None, weights=None)
        self.draining_size = draining_size
        self.batch_size = batch_size
        self.draining_margin = draining_margin

        # Yala Core attributes
        self.firing_graph = None
        self.server, self.drainer = None, None

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
        # encode X is a numpy/scipy array, y in numpy/scipy array
        X_enc, y_enc = self.encoder.fit_transform(X=X, y=y)

        # Instantiate core components
        if self.server_type == 'unclassified':
            self.server = YalaUnclassifiedServer(
                1, 0, X_enc, y_enc[:, 1], **dict(dropout_rate_mask=self.dropout_rate_mask)
            ).stream_features()
        elif self.server_type == 'misclassified':
            self.server = YalaMisclassifiedServer(
                1, 0, X_enc, y_enc[:, 1], **dict(dropout_rate_mask=self.dropout_rate_mask)
            ).stream_features()
        else:
            raise NotImplementedError

        # TODO: the procedure is as follow:
        #   Init: sample and init vertices
        #   Core:
        #       1. Refine: Refine sampled features and undiscovered one (add n_update)
        #       2. Expand: Expand sampled feature

        for i in range(self.max_iter):
            print("[YALA]: Iteration {}".format(i))

            # Initial sampling
            components = init_sample(self.n_node_by_iter, self.server, self.level_0, self.encoder.bf_map)

            # Core loop
            while True:

                # Refine
                fg, drainer_args = prepare_refinement(
                    partials, self.server, self.draining_size, 0.01, self.min_firing
                )

                drained = FiringGraphDrainer(fg, self.server, self.batch_size, **asdict(drainer_args.feedbacks)) \
                    .drain_all(n_max=self.draining_size) \
                    .firing_graph

                # Expand
                fg, drainer_args = prepare_expansion(
                    partials, self.server, self.draining_size, 0.01, self.min_firing
                )

                drained = FiringGraphDrainer(fg, self.server, self.batch_size, **asdict(drainer_args.feedbacks)) \
                    .drain_all(n_max=self.draining_size) \
                    .firing_graph

                # RE start from component and check stop criterion
                components, completes = None, None
                if True:
                    break

            # Augment current firing graph
            if self.firing_graph is not None:
                self.firing_graph = self.firing_graph.augment_from_fg_comp(completes)

            else:
                self.firing_graph = YalaBasePatterns.from_fg_comp(completes)

            # Update server
            if len(completes) > 0:
                self.server.update_mask_with_pattern(YalaBasePatterns.from_fg_comp(completes).reduce_output())
            self.server.sax_mask_forward = None
            self.server.pattern_backward = None

        import IPython
        IPython.embed()
        # TODO: use patterns to segment database and fit decision tree on each segment

        #analyze_patterns(self.firing_graph, mapping_feature_input, X, y)

        # test_amplifier = RefineAmplifier(
        #     self.server, mapping_feature_input, self.draining_size + 50000, min_size=self.min_firing,
        #     max_precision=self.max_precision, ci_value=0.5, gap_fill_length=2, cover_thresh=0.8,
        #     debug={"indices": [-6]}
        # )

        # fit logistic regression on patterns
        from sklearn.linear_model import LogisticRegression
        import itertools
        from src.mlops.names import KVName

        d_grid_params = {"penalty": ['l1', 'l2'], "C": [0.01, 0.1, 0.5, 1.0]}
        d_glob_params = {"fit_intercept": True}
        sax_x = self.firing_graph.propagate(X)
        for i, cross_values in enumerate(itertools.product(*d_grid_params.values())):
            d_search_params = d_glob_params.copy()
            d_search_params.update(dict(zip(d_grid_params.keys(), cross_values)))

            clf = LogisticRegression(**d_search_params)

            self.d_merger[KVName.from_dict(d_search_params).to_string()] = clf.fit(sax_x.A, y.A[:, 0])

        return self

    def predict_proba(self, X):

        assert self.firing_graph is not None, "First fit firing graph"

        sax_x = self.firing_graph.propagate(X)
        ax_probas = self.d_merger['C=0.1,fit_intercept=True,penalty=l2'].predict_proba(sax_x.A)
        return ax_probas[:, [1]]

    def predict(self, X, n_label, min_probas=0.1):
        pass