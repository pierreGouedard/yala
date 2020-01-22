# Global import
from firing_graph.core.tools.imputers import ArrayImputer
from firing_graph.core.solver.sampler import SupervisedSampler
from firing_graph.core.solver.drainer import FiringGraphDrainer
from firing_graph.core.data_structure.graph import FiringGraph

from scipy.sparse import lil_matrix
import numpy as np
import random

# Local import


class Yala(object):
    """
    Yala implement a Classifier algorithm based on the optimisation of firing graph (link paper). The classifier support
    multi-class classification and there is no restriction of the cardinality of each class. The Class implement
    the fundamentals of the BaseEstimator, BaseDecisionTree and ClassifierMixin interface from scikit-learn.
    In addition it implements model specific method of interest.

    """

    # TODO: Enable having p, q for each outputs and to store different precision value for each structure,
    # For this we need to modify fpo() which compute backward signals
    # TODO: Find a strategy to be able to select more than 1 bit at each iteration
    #

    def __init__(self,
                 max_cardinal_cat,
                 sampling_rate=0.8,
                 n_sampled_vertices=10,
                 max_iter=10,
                 learning_rate=5e-2,
                 p_flip=0.5,
                 batch_size=500,
                 firing_graph=None,
                 t=None,
                 min_firing=10
                 ):

        # Get category and numerical features
        self.max_cardinal_cat = max_cardinal_cat

        # Core parameter of the algorithm
        self.sampling_rate = sampling_rate
        self.n_sampled_vertices = n_sampled_vertices
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.p_flip = p_flip

        # Core attributes
        self.firing_graph = firing_graph
        self.drainer_params = {'t': t, 'min_firing': min_firing, 'batch_size': batch_size}
        self.min_firing = min_firing
        self.imputer = None
        self.sampler = None
        self.drainer = None

    def __feature_fit_transform(self, X):
        # TODO: encode using tools, infer categorical if necessary
        raise NotImplementedError

    def __feature_transform(self, X):
        # TODO: encode using tools, infer categorical if necessary
        raise NotImplementedError

    def __encode_target(self, y):
        # TODO: encode using tools, infer categorical if necessary
        raise NotImplementedError

    def __init_parameters(self, y):

        # Set batch sizefrom signal
        batch_size, t = min(y.shape[0], self.drainer_params['batch_size']), max(min(y.sum(axis=0)))

        # Set core params from signal and current firing graph
        init_precision = min(y.sum(axis=0) / y.shape[0])
        t_max = ((t / self.drainer_params['min_firing']) + 1) * y.shape[0]
        p, q = self.set_score_params(init_precision, init_precision * (1 - self.learning_rate))

        self.drainer_params.update({
            't_max': t_max, 'p': p, 'q': q, 'weight': (init_precision * (p + q) - p) * t, 't': t,
            'precision': init_precision, 'batch_size': batch_size
        })

    def __core_parameters(self, l_structures):

        # Set core params from signal and current firing graph
        precision = max(
            min([structure.precision for structure in l_structures]),
            self.drainer_params['precision']
        )
        p, q = self.set_score_params(precision, precision * (1 - self.learning_rate))

        self.drainer_params.update({
            'p': p, 'q': q, 'weight': (precision * (p + q) - p) * self.drainer_params['t'],
            'precision': precision * (1 - self.learning_rate)
        })

    def fit(self, X, y):

        # Sparse encode the features
        X = self.__feature_fit_transform(X)
        y = self.__encode_target(y)

        # Create and init imputer
        self.imputer = ArrayImputer(X, y)
        self.imputer.stream_features()

        # infer params from signal
        self.__init_parameters(y)

        # Init sampler
        self.sampler = SupervisedSampler(
            self.imputer, self.X.shape[1], y.shape[1], self.sampling_rate, self.n_sampled_vertices,
            firing_graph=self.firing_graph
        )

        for i in range(self.max_iter):
            print("[YALA]: Iteration {}".format(i))

            # Initial sampling
            firing_graph = self.sampler.generative_sampling().build_firing_graph(self.drainer_params)
            stop, n, is_output_flipped = False, 0, False

            # Core loop
            while not stop:

                # Drain firing graph
                firing_graph = FiringGraphDrainer(firing_graph, self.imputer)\
                    .drain_all(t_max=self.drainer_params['t_max'])\
                    .firing_graph

                # Augment firing graph with remaining samples
                l_structures, stop = self.augment_structures(firing_graph, is_output_flipped=is_output_flipped)

                if not stop:

                    is_output_flipped = random.random() < self.p_flip

                    # update parameters
                    self.__core_parameters(l_structures)

                    print("[YALA]: {} structures updated, targeted precision is {}".format(
                        len(l_structures), self.drainer_params['precision']
                    ))

                    # Update sampler
                    self.sampler.structures = l_structures

                    # Do discriminative sampling
                    firing_graph = self.sampler.discriminative_sampling().build_firing_graph(self.drainer_params)

                    n += 1
            # Merge firing graph
            self.sampler.merge_firing_graph(firing_graph)

        # Save partitioned firing graph
        self.firing_graph = self.sampler.firing_graph

    def augment_structures(self, firing_graph, is_output_flipped=False):

        l_structures, l_changed = [], []
        for partition in firing_graph.partitions:
            if partition.get('partitions', None) is not None:
                l_sub_structures, changed = self.create_base_structures(partition, firing_graph, is_output_flipped)

            else:
                l_sub_structures, changed = self.update_base_structures(partition, firing_graph, is_output_flipped)

            l_structures.extend(l_sub_structures)
            l_changed.append(changed)

        return l_structures, not any(l_changed)

    def create_base_structures(self, partition, firing_graph, is_output_flipped):

        # Get information on structure
        index_output = (firing_graph.O[partition['indices'], :].sum(axis=0) > 0).nonzero()[1][0]
        n_inputs, n_outputs = firing_graph.I.shape[0], firing_graph.O.shape[1]

        # Build structure for each drained vertex of the partition
        l_selected, l_structures = [], []

        for ind in partition['indices']:
            d_candidate = self.select_best_candidate(
                firing_graph.Im[:, ind], firing_graph.Iw[:, ind], l_selected, self.drainer_params
            )

            if d_candidate is None:
                continue

            l_structures.append(self.create_base_structure(
                index_output, d_candidate, n_inputs, n_outputs, is_output_flipped=is_output_flipped
            ))
            l_selected.append([d_candidate['index']])

        return l_structures, len(l_structures) > 1

    def update_base_structures(self, partition, firing_graph, is_output_flipped):

        l_structures, base_structure = [], None

        # Check whether there is a base and transient structure
        self.check_sub_partitions(partition['partitions'])

        # Extract both su partition
        for sub_partition in partition['partitions']:
            l_keys = list(sub_partition['indices'])

            if sub_partition['name'] == 'base':
                base_partition = {
                    'indices': [partition['indices'][k] for k in l_keys],
                    'depth': sub_partition['depth']
                }
                base_structure = self.extract_structure(base_partition, firing_graph)

            else:
                transient_partition = {
                    'indices': [partition['indices'][k] for k in l_keys],
                    'depth': sub_partition['depth']
                }

        # Update base structure with selected transient sructure drained bits
        l_selected = []
        for ind in transient_partition['indices']:
            d_candidate = self.select_best_candidate(
                firing_graph.Im[:, ind], firing_graph.Iw[:, ind], l_selected, self.drainer_params,
                precision_treshold=partition['precision'], l_base_indices=base_partition['indices']
            )

            if d_candidate is None:
                continue

            l_structures.append(self.update_base_structure(
                d_candidate, base_structure, is_output_flipped=is_output_flipped
            ))
            l_selected.append(base_partition['indices'] + [d_candidate['index']])

        if len(l_structures) == 0:
            return [base_structure], False

        return l_structures, True

    @staticmethod
    def select_best_candidate(sax_mask, sax_scores, l_selected, drainer_params, precision_treshold=0,
                              l_base_indices=None):
        if sax_scores.nnz > 0:
            l_candidates = [c for c in (sax_scores > 0).nonzero()[0] if not sax_mask[c, 0]]
            l_candidates = zip(l_candidates, [sax_scores[c, 0] for c in l_candidates])

            for c, c_score in sorted(l_candidates, key=lambda t: t[1], reverse=True):
                if l_base_indices is not None:
                    l_indices = l_base_indices.append(c)
                else:
                    l_indices = [c]

                if l_indices not in l_selected:
                    precision = float(c_score - drainer_params['weight'])
                    precision /= (drainer_params['t'] * (drainer_params['p'] + drainer_params['q']))
                    precision += float(drainer_params['p']) / (drainer_params['p'] + drainer_params['q'])

                    if precision > precision_treshold:
                        return {'index': c, 'precision': precision}
        return

    @staticmethod
    def check_sub_partitions(d_partitions):
        try:
            sub_partition_names = [part.get('name', None) for part in d_partitions]
            assert('transient' in sub_partition_names and 'base' in sub_partition_names)

        except AssertionError:
            raise ValueError("Either 'base' or 'transient' part of the firing graph missing")

    @staticmethod
    def set_score_params(phi_old, phi_new, q_max=1000):
        for q in range(q_max):
            p = np.ceil(q * phi_old / (1 - phi_old))

            score = (phi_new * (p + q)) - p

            if score > 0.:
                return p, q

    @staticmethod
    def create_base_structure(index_output, d_candidate, n_inputs, n_outputs, is_output_flipped=False):

        # Initialize matrices
        sax_I = lil_matrix((n_inputs, 5))
        sax_C = lil_matrix((5, 5))
        sax_O = lil_matrix((5, n_outputs))

        if is_output_flipped:
            sax_I[d_candidate['index'], 1] = 1
        else:
            sax_I[d_candidate['index'], 0] = 1

        ax_levels = np.array([sax_I[:, 0].nnz, sax_I[:, 1].nnz, 1, 1, 2])

        sax_C[0, 2], sax_C[0, 3], sax_C[1, 3], sax_C[3, 4], sax_C[2, 4] = 1, 1, 1, 1, 1
        sax_O[4, index_output] = 1

        d_masks = {
            'Im': lil_matrix(sax_I.shape, dtype=bool),
            'Cm': lil_matrix(sax_C.shape, dtype=bool),
            'Om': lil_matrix(sax_O.shape, dtype=bool)
        }
        return FiringGraph.from_matrices(
            sax_I.tocsc(), sax_C.tocsc(), sax_O.tocsc(), ax_levels, mask_matrices=d_masks, depth=4,
            precision=d_candidate['precision']
        )

    @staticmethod
    def update_base_structure(d_candidate, structure, is_output_flipped=False):

        # Initialize matrices
        sax_I, sax_C, sax_O = structure.Iw.tolil(), structure.Cw.tolil(), structure.Ow.tolil()
        if is_output_flipped:
            sax_I[d_candidate['index'], 1] = 1

        else:
            sax_I[d_candidate['index'], 0] = 1

        ax_levels = np.array([sax_I[:, 0].nnz, sax_I[:, 1].nnz, 1, 1, 2])
        d_masks = {
            'Im': lil_matrix(sax_I.shape, dtype=bool),
            'Cm': lil_matrix(sax_C.shape, dtype=bool),
            'Om': lil_matrix(sax_O.shape, dtype=bool)
        }
        return FiringGraph.from_matrices(
            sax_I.tocsc(), sax_C.tocsc(), sax_O.tocsc(), ax_levels, mask_matrices=d_masks, depth=4,
            precision=d_candidate['precision']
        )

    @staticmethod
    def extract_structure(partition, firing_graph):
        """

        :param partition:
        :param firing_graph:
        :return:
        """
        l_ind_partition = partition['indices']

        sax_I = firing_graph.Iw[:, l_ind_partition]
        sax_C = firing_graph.Cw[l_ind_partition, :][:, l_ind_partition]
        sax_O = firing_graph.Ow[l_ind_partition, :]

        d_masks = {
            'Im': firing_graph.Im[:, l_ind_partition],
            'Cm': firing_graph.Cm[l_ind_partition, :][:, l_ind_partition],
            'Om': firing_graph.Om[l_ind_partition, :]
        }

        ax_levels = firing_graph.levels[l_ind_partition]

        return FiringGraph.from_matrices(
            sax_I.tocsc(), sax_C.tocsc(), sax_O.tocsc(), ax_levels, mask_matrices=d_masks, depth=partition['depth'],
            partitions=partition.get('partitions', None), precision=partition.get('precision', None)
        )

    def predict(self, X):
        ax_probas = self.predict_probas(X)
        return ax_probas - ax_probas.max(axis=1) >= -1e-5

    def predict_probas(self, X):
        assert self.firing_graph is not None, "First fit firing graph"

        # transform features
        X = self.__feature_transform(X)

        ax_probas = np.zeros(self.firing_graph.O.shape[1])
        for partition in self.firing_graph.partitions:
            structure = self.extract_structure(partition, self.firing_graph)
            sax_probas = structure.propagate(X).multiply(structure.precision)

            for i, j in zip(sax_probas.nonzero()):
                if ax_probas[i, j] < sax_probas[i, j]:
                    ax_probas[i, j] = structure.precision

        return ax_probas

    def score(self, X, y):
        raise NotImplementedError
