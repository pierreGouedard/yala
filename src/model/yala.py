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
                 sampling_rate=0.8,
                 n_sampled_vertices=10,
                 max_iter=5,
                 learning_rate=2e-5,
                 p_flip=0.5,
                 batch_size=500,
                 firing_graph=None,
                 t=None,
                 min_firing=10,
                 max_precision=0.9,
                 treshold_precision=0.5,
                 overlap_rate=0.5
                 ):

        # Core parameter of the algorithm
        self.sampling_rate = sampling_rate
        self.n_sampled_vertices = n_sampled_vertices
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.p_flip = p_flip
        self.max_precision = max_precision
        self.treshold_precision = treshold_precision
        self.overlap_rate = overlap_rate
        self.batch_size = batch_size

        # Core attributes
        self.firing_graph = firing_graph
        self.drainer_params = {'t': t, 'min_firing': min_firing, 'batch_size': batch_size}
        self.min_firing = min_firing
        self.n_outputs = None
        self.n_inputs = None
        self.imputer = None
        self.sampler = None
        self.drainer = None

    def __init_parameters(self, y):

        # Set batch sizefrom signal
        t, t_max = y.shape[0], y.shape[0]

        # Set core params from signal and current firing graph
        init_precision = np.array(y.sum(axis=0) / y.shape[0]).min()
        precision = init_precision * (1 + self.learning_rate)

        # Get scoring process params
        p, q = self.set_score_params(init_precision, precision)

        self.drainer_params.update({
            't_max': t_max, 'p': p, 'q': q, 'weight': (p - init_precision * (p + q)) * self.min_firing, 't': t,
            'precision': init_precision
        })

    def __core_parameters(self, l_structures):

        # Set core params from signal and current firing graph
        min_precision = min([structure.precision for structure in l_structures if structure.precision is not None])
        init_precision = max(min_precision, self.drainer_params['precision'])
        precision = init_precision * (1 + self.learning_rate)

        if precision > self.max_precision:
            init_precision, precision = self.max_precision - 5e-2, self.max_precision

        p, q = self.set_score_params(init_precision, precision)

        self.drainer_params.update({
            'p': p, 'q': q, 'weight': int((p - init_precision * (p + q)) * self.min_firing) + 1, 'precision': precision
        })

    def fit(self, X, y, update=False):

        # Initialisation
        if not update:
            self.firing_graph = None

        self.imputer = ArrayImputer(X, y)
        self.imputer.stream_features()

        self.n_inputs, self.n_outputs = X.shape[1], y.shape[1]
        self.batch_size = min(y.shape[0], self.batch_size)

        self.sampler = SupervisedSampler(
            self.imputer, X.shape[1], y.shape[1], self.batch_size, self.sampling_rate, self.n_sampled_vertices,
            firing_graph=self.firing_graph
        )

        # Core loop
        for i in range(self.max_iter):
            print("[YALA]: Iteration {}".format(i))

            # infer params from signal
            self.__init_parameters(y)

            # Initial sampling
            firing_graph = self.sampler.generative_sampling().build_firing_graph(self.drainer_params)
            stop, n = False, 0
            self.imputer.set_backward_flip(False)

            # Core loop
            while not stop:

                # Drain firing graph
                firing_graph = FiringGraphDrainer(firing_graph, self.imputer, self.batch_size)\
                    .drain_all(t_max=self.drainer_params['t_max'])\
                    .firing_graph

                # Augment firing graph with remaining samples
                l_structures, n_updates = self.augment_structures(
                    firing_graph, X, y, is_output_flipped=self.imputer.backward_flip
                )

                # Compute stop criteria
                stop = (n_updates == 0)

                if not stop:

                    self.imputer.set_backward_flip(random.random() < self.p_flip)

                    # update parameters
                    self.__core_parameters(l_structures)

                    print("[YALA]: {} structures updated, targeted precision is {}".format(
                        n_updates, self.drainer_params['precision']
                    ))

                    # Sample
                    self.sampler.structures = l_structures
                    firing_graph = self.sampler.discriminative_sampling().build_firing_graph(self.drainer_params)

                    n += 1

                else:
                    firing_graph = self.sampler.merge_structures(l_structures, drainer_params=None)

            # Merge firing graph
            if n != 0:
                self.sampler.merge_firing_graph(firing_graph)

            self.sampler.structures = None

        # Save partitioned firing graph
        self.firing_graph = self.sampler.firing_graph

        return self

    def augment_structures(self, firing_graph, X, y, is_output_flipped=False):

        l_structures, n_updates = [], 0
        for i in range(self.n_outputs):
            l_partition_sub = [partition for partition in firing_graph.partitions if partition['output'] == i]
            l_sub_structures, n = self.update_structures(l_partition_sub, firing_graph, i, is_output_flipped, X, y)

            l_structures.extend(l_sub_structures)
            n_updates += n

        return l_structures, n_updates

    def update_structures(self, l_partitions, firing_graph, ind_out, is_output_flipped, X, y):

        l_structures, n, ax_signal = [], 0, np.zeros(X.shape[0], dtype=bool)
        ax_y = y.toarray()[:, 0].astype(int)
        for partition in l_partitions:

            # Extract partition
            firing_graph_sub, l_structure_sub = self.extract_structure(partition, firing_graph), []

            # For each sub partition
            d_sub_partitions = {sub_part['name']: sub_part for sub_part in firing_graph_sub.partitions}
            base_structure = self.extract_structure(d_sub_partitions['base'], firing_graph_sub, index_output=ind_out)

            for ind in d_sub_partitions['transient']['indices'][:-1]:
                for d_candidate in self.get_candidates(firing_graph_sub, ind, partition.get('precision', 0)):

                    # Update structure
                    if base_structure is not None:
                        structure = self.update_base_structure(
                            ind_out, d_candidate, base_structure, is_output_flipped=is_output_flipped
                        )
                    else:
                        structure = self.create_base_structure(
                            ind_out, d_candidate, self.n_inputs, self.n_outputs, is_output_flipped=False
                        )

                    # test for overlap
                    ax_structure = structure.propagate(X).toarray()[:, 0] * ax_y
                    if ax_signal.astype(int).dot(ax_structure) < self.overlap_rate * ax_structure.sum():
                        ax_signal = (ax_signal + ax_structure) > 0
                        l_structure_sub.append(structure)
                        n += 1
                        break

            if len(l_structure_sub) == 0 and base_structure is not None:
                l_structure_sub.append(base_structure)

            l_structures.extend(l_structure_sub)

        return l_structures, n

    def get_candidates(self, firing_graph, ind, precision_treshold=0):

        # Get quantity of interest
        sax_scores, sax_t = firing_graph.Iw[:, ind], firing_graph.backward_firing['i'].tocsc()[:, ind]

        if sax_scores.nnz > 0:
            l_indices = [i for i in (sax_scores > 0).nonzero()[0] if sax_t[i, 0] > self.min_firing]
            l_precisions = [self.get_precision(sax_scores[i, 0], sax_t[i, 0]) for i in l_indices]

            for ind, precision in sorted(zip(l_indices, l_precisions), key=lambda t: t[1], reverse=True):
                if precision > precision_treshold:
                    yield {'index': ind, 'precision': precision}

    def get_precision(self, score, t):
        precision = float(score - self.drainer_params['weight'])
        precision /= (t * (self.drainer_params['p'] + self.drainer_params['q']))
        precision += float(self.drainer_params['p']) / (self.drainer_params['p'] + self.drainer_params['q'])
        return precision

    @staticmethod
    def set_score_params(phi_old, phi_new, q_max=1000):

        for q in range(q_max):
            p = np.ceil(q * phi_old / (1 - phi_old))

            score = (phi_new * (p + q)) - p
            if score > 0.:
                print(p, q)
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

        ax_levels = np.array([max(sax_I[:, 0].nnz, 1), max(sax_I[:, 1].nnz, 1), 1, 1, 2])

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
    def update_base_structure(index_output, d_candidate, structure, is_output_flipped=False):

        # Initialize matrices
        sax_I, sax_C, sax_O = structure.Iw.tolil(), structure.Cw.tolil(), structure.Ow.tolil()

        if is_output_flipped:
            sax_I[d_candidate['index'], 1] = 1

        else:
            sax_I[d_candidate['index'], 0] = 1

        ax_levels = np.array([max(sax_I[:, 0].nnz, 1), max(sax_I[:, 1].nnz, 1), 1, 1, 2])
        d_masks = {
            'Im': lil_matrix(sax_I.shape, dtype=bool),
            'Cm': lil_matrix(sax_C.shape, dtype=bool),
            'Om': lil_matrix(sax_O.shape, dtype=bool)
        }

        sax_O[4, index_output] = 1

        return FiringGraph.from_matrices(
            sax_I.tocsc(), sax_C.tocsc(), sax_O.tocsc(), ax_levels, mask_matrices=d_masks, depth=4,
            precision=d_candidate['precision']
        )

    @staticmethod
    def extract_structure(partition, firing_graph, index_output=None, precision=None):
        """

        :param partition:
        :param firing_graph:
        :return:
        """

        l_ind_partition = partition['indices']

        if len(l_ind_partition) == 0:
            return None

        sax_I = firing_graph.Iw.tocsc()[:, l_ind_partition]
        sax_C = firing_graph.Cw.tocsc()[l_ind_partition, :][:, l_ind_partition]
        sax_O = firing_graph.Ow.tocsc()[l_ind_partition, :].tolil()

        d_masks = {
            'Im': firing_graph.Im.tocsc()[:, l_ind_partition].tocoo(),
            'Cm': firing_graph.Cm.tocsc()[l_ind_partition, :][:, l_ind_partition].tocoo(),
            'Om': firing_graph.Om.tocsc()[l_ind_partition, :].tocoo()
        }

        d_backward_firing = {
            'i': firing_graph.backward_firing['i'].tocsc()[:, l_ind_partition],
            'c': firing_graph.backward_firing['c'].tocsc()[:, l_ind_partition][l_ind_partition, :],
            'o': firing_graph.backward_firing['o'].tocsc()[l_ind_partition, :],
        }

        if index_output is not None:
            sax_O[4, index_output] = 1

        ax_levels = firing_graph.levels[l_ind_partition]

        firing_graph_sub = FiringGraph.from_matrices(
            sax_I, sax_C, sax_O.tocsc(), ax_levels, mask_matrices=d_masks, depth=partition['depth'],
            partitions=partition.get('partitions', None), precision=partition.get('precision', precision)
        )

        firing_graph_sub.backward_firing = d_backward_firing

        return firing_graph_sub

    def predict(self, X):
        ax_probas = self.predict_probas(X)

        if ax_probas.shape[1] == 1:
            ax_preds = (ax_probas[:, 0] > self.treshold_precision).astype(int)

        else:
            ax_preds = ax_probas.argmax(axis=1)
        return ax_preds

    def predict_probas(self, X):
        assert self.firing_graph is not None, "First fit firing graph"

        ax_probas = np.zeros((X.shape[0], self.firing_graph.O.shape[1]))
        for partition in self.firing_graph.partitions:
            structure = self.extract_structure(partition, self.firing_graph)
            sax_probas = structure.propagate(X).multiply(structure.precision)
            for i, j in zip(*sax_probas.nonzero()):
                if ax_probas[i, j] < sax_probas[i, j]:
                    ax_probas[i, j] = structure.precision

        return ax_probas

    def score(self, X, y):
        raise NotImplementedError
