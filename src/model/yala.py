# Global import
from firing_graph.core.tools.imputers import ArrayImputer
from firing_graph.core.solver.sampler import SupervisedSampler
from firing_graph.core.solver.drainer import FiringGraphDrainer
from firing_graph.core.data_structure.structures import StructureIntersection, StructureYala
import numpy as np

# Local import


class Yala(object):
    """
    Yala implement a Classifier algorithm based on the optimisation of firing graph (link paper). The classifier support
    multi-class classification and there is no restriction of the cardinality of each class. The Class implement
    the fundamentals of the BaseEstimator, BaseDecisionTree and ClassifierMixin interface from scikit-learn.
    In addition it implements model specific method of interest.

    """

    # TODO: Enable having p, q for each outputs and to store different precision value for each structure,
    # TODO: Find a strategy to be able to select more than 1 bit at each iteration
    # TODO: Enable the use of streamer, how does it affect things ?

    def __init__(self,
                 sampling_rate=0.8,
                 n_sampled_vertices=10,
                 max_iter=5,
                 learning_rate=5e-2,
                 batch_size=1000,
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
        self.max_precision = max_precision
        self.treshold_precision = treshold_precision
        self.overlap_rate = overlap_rate
        self.batch_size = batch_size

        # Core attributes
        self.firing_graph = firing_graph
        self.drainer_params = {'t': t, 'min_firing': min_firing, 'batch_size': batch_size}
        self.min_firing = min_firing
        self.n_outputs, self.n_inputs = None, None
        self.imputer, self.sampler, self.drainer = None, None, None

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

            # Core loop
            while not stop:

                # Drain firing graph
                firing_graph = FiringGraphDrainer(firing_graph, self.imputer, self.batch_size)\
                    .drain_all(t_max=self.drainer_params['t_max'])\
                    .firing_graph

                # Augment firing graph with remaining samples
                l_structures, n_updates = self.augment_structures(firing_graph, X, y)

                # Compute stop criteria
                stop = (n_updates == 0)

                if not stop:

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
            structure = StructureIntersection.from_partition(partition, self.firing_graph)
            sax_probas = structure.propagate(X).multiply(structure.precision)
            for i, j in zip(*sax_probas.nonzero()):
                if ax_probas[i, j] < sax_probas[i, j]:
                    ax_probas[i, j] = structure.precision

        return ax_probas

    def score(self, X, y):
        raise NotImplementedError

    def augment_structures(self, firing_graph, X, y):

        l_structures, n_updates = [], 0
        for i in range(self.n_outputs):
            l_partition_sub = [partition for partition in firing_graph.partitions if partition['index_output'] == i]
            l_sub_structures, n = self.update_structures(l_partition_sub, firing_graph, X, y)

            l_structures.extend(l_sub_structures)
            n_updates += n
        return l_structures, n_updates

    def update_structures(self, l_partitions, firing_graph, X, y):

        l_structures, n, ax_signal, ax_y = [], 0, np.zeros(X.shape[0], dtype=bool), y.toarray()[:, 0].astype(int)

        for partition in l_partitions:

            # Init selected structure list
            l_structure_sub = []

            # Extract yala sampling structure
            sampled_structure = StructureYala.from_partition(partition, firing_graph, add_backward_firing=True)

            # Get sub partition (base and transient)
            d_sub_partitions = {sub_part['name']: sub_part for sub_part in sampled_structure.partitions}

            # Extract base partition
            base_structure = StructureIntersection.from_partition(
                d_sub_partitions['base'], sampled_structure, index_output=partition['index_output'],
                add_backward_firing=True
            )
            for ind in d_sub_partitions['transient']['indices']:
                for d_bit in self.get_scored_bits(sampled_structure, ind, partition.get('precision', 0)):

                    # Update structure
                    if base_structure is not None:
                        structure = base_structure.copy().augment_intersection([d_bit['index']], 1, delta_level=1)
                        structure.precision = d_bit['precision']

                    else:
                        structure = StructureIntersection.from_input_indices(
                            self.n_inputs, self.n_outputs, 1, partition['index_output'], [[d_bit['index']]], 1,
                            enable_drain=False, **{'precision': d_bit['precision']}
                        )

                    # Validate overlapping rate
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

    def get_scored_bits(self, firing_graph, ind, precision_treshold=0):

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
