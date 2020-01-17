# Global import
from firing_graph.core.tools.imputers import ArrayImputer
from firing_graph.core.solver.sampler import SupervisedSampler
from firing_graph.core.solver.drainer import FiringGraphDrainer
from firing_graph.core.data_structure.graph import FiringGraph, extract_structure

from scipy.sparse import csc_matrix, lil_matrix, hstack, vstack
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

    def __init__(self, sampling_rate=0.8, n_sampled_vertices=10, max_iter=10, learning_rate=5e-2,
                 p_flip=0.5, batch_size=500, firing_graph=None, t=None, min_firing=10):

        # Core parameter of the algorithm
        self.sampling_rate = sampling_rate
        self.n_sampled_vertices = n_sampled_vertices
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.p_flip = p_flip
        self.batch_size = batch_size

        # Core attributes
        self.firing_graph = firing_graph
        self.t = t
        self.min_firing = min_firing
        self.imputer = None
        self.sampler = None
        self.drainer = None

    def __init_parameters(self, y, infer_time=True):
        # Set batch sizefrom signal
        self.batch_size = min(y.shape[0], self.batch_size)
        self.t = max(min(y.sum(axis=0)))
        t_max = ((self.t / self.min_firing) + 1) * y.shape[0]

        # Set core params from signal and current firing graph
        init_precision = min(y.sum(axis=0) / y.shape[0])

        p, q = self.set_score_params(init_precision - 1e-1, init_precision)
        ax_weight = (init_precision * (p + q) - p) * self.t * (p + q)

        return t_max, p, q, ax_weight, init_precision

    def __core_parameters(self, init_precision, step, l_structures):

        # Set core params from signal and current firing graph
        precision = init_precision * pow(self.learning_rate, step)
        p, q = self.set_score_params(
            init_precision * pow(self.learning_rate, step - 1), init_precision * pow(self.learning_rate, step)
        )
        ax_weight = np.ones(len(l_structures)) * ((precision * (p + q) - p) * (self.t * (p + q)))

        return p, q, ax_weight

    def fit(self, X, y, infer_time=True):
        # TODO: investigate the possibility to make different p, q for each vertex and to store different precision value for each structure
        # TODO: Its possible in the case where each structure is independant at a deeper layer than input (no core has path toward different
        # TODO: different output. It must be done in the function fpo() where p and q can be replaced by verctors, then a broadcasted
        # TODO: element wise multiplication can be done. May be in a V0
        # TODO: For the precision, may be an estimation of precision of each structure of the firing graph ma be done
        # TODO: outside the core loop of draining and sampling. may be a partition
        # Create and init imputer
        self.imputer = ArrayImputer(X, y)
        self.imputer.stream_features()

        # infer params from signal
        t_max, p, q, ax_weight, ax_precision_outputs = self.__init_parameters(y, infer_time=infer_time)

        # Init sampler
        self.sampler = SupervisedSampler(
            self.imputer, self.X.shape[1], y.shape[1], self.sampling_rate, self.n_sampled_vertices,
            firing_graph=self.firing_graph
        )

        for i in range(self.max_iter):

            # Initial sampling
            firing_graph = self.sampler.generative_sampling().build_firing_graph(ax_weight)
            stop, n, is_output_flipped = False, 0, False

            # Core loop
            while not stop:

                # Drain firing graph
                firing_graph = FiringGraphDrainer(self.t, p, q, self.batch_size, firing_graph, self.imputer)\
                    .drain_all(t_max=t_max)\
                    .firing_graph

                # Augment firing graph with remaining samples
                l_structures, stop = self.augment_structures(firing_graph, is_output_flipped=is_output_flipped)

                if not stop:

                    is_output_flipped = random.random() < self.p_flip

                    # update parameters
                    t, p, q, ax_weight = self.__core_parameters(ax_precision_outputs, n, l_structures)

                    # Update sampler
                    self.sampler.structures = l_structures

                    # Do discriminative sampling
                    firing_graph = self.sampler.discriminative_sampling().build_firing_graph(ax_weight)

                    n += 1

            self.sampler.firing_graph = firing_graph

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
            l_indices = self.select_best_candidate(firing_graph.backward_firing['i'], ind, l_selected)

            if l_indices is None:
                continue

            l_structures.append(self.create_base_structure(
                index_output, l_indices, n_inputs, n_outputs, is_output_flipped=is_output_flipped
            ))
            l_selected.extend(l_indices)

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
                base_structure = extract_structure(base_partition, firing_graph)

            else:
                transient_partition = {
                    'indices': [partition['indices'][k] for k in l_keys],
                    'depth': sub_partition['depth']
                }

        # Update base structure with selected transient sructure drained bits
        l_selected = []
        for ind in transient_partition['indices']:
            l_indices = self.select_best_candidate(firing_graph.backward_firing['i'], ind, l_selected)

            if l_indices is None:
                continue

            l_structures.append(self.update_base_structure(
                l_indices, base_structure, is_output_flipped=is_output_flipped
            ))
            l_selected.extend(l_indices)

        if len(l_structures) == 0:
            return [base_structure], False

        return l_structures, True

    @staticmethod
    def select_best_candidate(sax_backward_firing, ind, l_selected, l0=1):
        # TODO: handle the case l0 > 1 so that we want to sample more than one (Not in V0)
        purity_scores = sax_backward_firing[:, ind]
        if purity_scores.nnz > 0:

            l_candidates = list((purity_scores > 0).nonzero()[0])
            l_candidates = zip(l_candidates, [purity_scores[c, 0] for c in l_candidates])

            for c, c_score in sorted(l_candidates, key=lambda t: t[1], reverse=True):
                if c not in l_selected:
                    return [c]

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

    def create_base_structure(self, index_output, l_indices, n_inputs, n_outputs, is_output_flipped=False):

        # Initialize matrices
        sax_I = lil_matrix((n_inputs, 5))
        sax_C = lil_matrix((5, 5))
        sax_O = lil_matrix((5, n_outputs))

        if is_output_flipped:
            sax_I[l_indices, 1] = 1
        else:
            sax_I[l_indices, 0] = 1

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
        )

    def update_base_structure(self, l_indices, structure, is_output_flipped=False):

        # Initialize matrices
        sax_I, sax_C, sax_O = structure.Iw.tolil(), structure.Cw.tolil(), structure.Ow.tolil()
        ax_level = structure.levels

        if is_output_flipped:
            sax_I[l_indices, 1] = 1

        else:
            sax_I[l_indices, 0] = 1

        ax_levels = np.array([sax_I[:, 0].nnz, sax_I[:, 1].nnz, 1, 1, 2])
        d_masks = {
            'Im': lil_matrix(sax_I.shape, dtype=bool),
            'Cm': lil_matrix(sax_C.shape, dtype=bool),
            'Om': lil_matrix(sax_O.shape, dtype=bool)
        }
        return FiringGraph.from_matrices(
            sax_I.tocsc(), sax_C.tocsc(), sax_O.tocsc(), ax_levels, mask_matrices=d_masks, depth=4,
        )

    def predict(self, X, y):
        raise NotImplementedError

    def predict_probas(self, X, y):
        raise NotImplementedError

    def score(self, X, y):
        raise NotImplementedError
