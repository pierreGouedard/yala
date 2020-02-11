# Global import
from firing_graph.core.data_structure.graph import FiringGraph
from firing_graph.core.data_structure.utils import create_empty_matrices, augment_matrices

from scipy.sparse import csc_matrix
import numpy as np
import unittest

# Local import
from src.model.patterns import YalaBasePattern, YalaTransientPattern, YalaSingleDrainingPattern


class TestPattern(unittest.TestCase):

    def setUp(self):

        self.n_inputs, self.n_outputs, self.index_outputs, self.n_rand = 10, 2, np.random.randint(0, 2), 2
        self.transient_index, self.base_index = [[0, 1], [2, 3]], [7, 8, 9]
        self.depth = 4

        # build test signal
        self.X_base = csc_matrix(np.vstack((
            np.hstack((np.ones(7), np.zeros(3))),
            np.hstack((np.ones(8), np.zeros(2))),
            np.hstack((np.ones(9), np.zeros(1))),
            np.hstack((np.ones(10), np.zeros(0)))
        )))
        self.y_base = np.array([[False], [False], [False], [True]])

        self.X_transient = csc_matrix(np.vstack((
            np.hstack((np.zeros(4), np.ones(6))),
            np.hstack((np.zeros(3), np.ones(7))),
            np.hstack((np.zeros(2), np.ones(8))),
            np.hstack((np.zeros(1), np.ones(9))),
            np.hstack((np.zeros(0), np.ones(10)))
        )))
        self.y_transient = np.array([[False], [True], [True], [True], [True]])

        # Create GOT matrices
        self.got_matrices = create_empty_matrices(self.n_inputs, self.n_outputs, 6)

        self.got_matrices['Iw'] = csc_matrix(np.hstack((
            np.vstack((np.zeros((7, 1)), np.ones((3, 1)))),
            np.vstack((np.ones((2, 1)), np.zeros((8, 1)))),
            np.vstack((np.zeros((2, 1)), np.ones((2, 1)), np.zeros((6, 1)))),
            np.zeros((10, 3))
        )))
        self.got_matrices['Cw'] = csc_matrix(np.vstack((
            np.array([0, 0, 0, 0, 1, 0]),
            np.array([0, 0, 0, 1, 0, 0]),
            np.array([0, 0, 0, 1, 0, 0]),
            np.array([0, 0, 0, 0, 0, 1]),
            np.array([0, 0, 0, 0, 0, 1]),
            np.array([0, 0, 0, 0, 0, 0]),
        )))
        self.got_matrices['Ow'][-1, self.index_outputs] = 1
        self.ax_got_levels = np.array([3, 1, 1, 1, 1, 2])

        # Create random part and build test firing graph
        matrices_rand_1, ax_rand_levels = create_random_matrices(self.n_inputs, self.n_outputs, self.n_rand, depth=4)
        matrices_final = augment_matrices(matrices_rand_1, self.got_matrices)
        ax_levels_final = np.hstack((ax_rand_levels, self.ax_got_levels))

        self.firing_graph = FiringGraph('testPattern', ax_levels_final, matrices_final, depth=4, partitions=None)

        # Initialize firing backward to random int
        self.ax_backward_transient = np.empty_like([], shape=(0, len(self.transient_index[0])))
        self.firing_graph.backward_firing['i'] = self.firing_graph.backward_firing['i'].tolil()
        for i, l_inds in enumerate(self.transient_index):
            ax_firing = np.random.randint(10, 21, len(l_inds))
            self.firing_graph.backward_firing['i'][l_inds, i + self.n_rand + 1] = ax_firing
            self.ax_backward_transient = np.vstack((self.ax_backward_transient, ax_firing))
        self.firing_graph.backward_firing['i'] = self.firing_graph.backward_firing['i'].tocsc()

    def test_base_pattern(self):
        """
        python -m unittest tests.units.test_patterns.TestPattern.test_base_pattern

        """

        # Test base pattern from indices
        pattern = YalaBasePattern.from_input_indices(
            self.n_inputs, self.n_outputs, self.index_outputs, self.base_index
        )
        self.assertion_base(pattern)

        # Test base pattern from partition of a larger firing graph
        partition_base = {
            'indices': [self.n_rand + self.depth - 2],
            'precision': 0.1234
        }
        pattern = YalaBasePattern.from_partition(partition_base, self.firing_graph, self.index_outputs)

        self.assertion_base(pattern)
        self.assertEqual(pattern.precision, partition_base['precision'])

        # Test augment base pattern
        pattern.augment([0])
        self.assertion_base(pattern, offset=1, signal_assert=False)

    def test_transient_pattern(self):
        """
        python -m unittest tests.units.test_patterns.TestPattern.test_transient_pattern

        """

        # test transient pattern from indices
        pattern = YalaTransientPattern.from_input_indices(
            self.n_inputs, self.n_outputs, 1, self.index_outputs, self.transient_index, 1
        )
        self.assertion_transient(pattern)

        # test transient pattern from a larger graph
        partition_transient = {
            'indices': [i + self.n_rand + self.depth - 1 for i in range(len(self.transient_index) + 1)]
        }

        pattern = YalaTransientPattern.from_partition(
            partition_transient, self.firing_graph, self.index_outputs, add_backward_firing=True
        )

        self.assertion_transient(pattern, check_backward=True)

    def test_draining_pattern(self):
        """
        python -m unittest tests.units.test_patterns.TestPattern.test_draining_pattern

        """
        # Test creation of single draining pattern
        base_pattern = YalaBasePattern.from_input_indices(
            self.n_inputs, self.n_outputs, self.index_outputs, self.base_index
        )

        transient_pattern = YalaTransientPattern.from_input_indices(
            self.n_inputs, self.n_outputs, 1, self.index_outputs, self.transient_index, 1
        )

        draining_pattern = YalaSingleDrainingPattern.from_patterns(base_pattern, transient_pattern)

        # Check matrices
        self.assertTrue((self.got_matrices['Iw'].toarray() == draining_pattern.Iw.toarray()).all())
        self.assertTrue((draining_pattern.Cw.toarray() == self.got_matrices['Cw'].toarray()).all())
        self.assertTrue((draining_pattern.Ow.toarray() == self.got_matrices['Ow'].toarray()).all())
        self.assertTrue((draining_pattern.levels == self.ax_got_levels).all())

    # def test_predicting_pattern(self):
    #     """
    #     python -m unittest tests.units.test_patterns.TestPattern.test_transient_pattern
    #
    #     """
    #     # Test creation of single draining pattern
    #
    #     # test creation of a multiple draining pattern
    #
    #     raise NotImplementedError

    def assertion_base(self, base_pattern, signal_assert=True, offset=0):

        # Check structure dims
        self.assertEqual(base_pattern.depth, 2)
        self.assertEqual(base_pattern.I.shape, (self.n_inputs, 1))
        self.assertEqual(base_pattern.C.shape, (1, 1))
        self.assertEqual(base_pattern.O.shape, (1, self.n_outputs))
        self.assertEqual(base_pattern.I.nnz, len(self.base_index) + offset)
        self.assertEqual(len(base_pattern.levels), 1)
        self.assertEqual(base_pattern.levels[0], len(self.base_index) + offset)
        self.assertEqual(base_pattern.O.nnz, 1)

        # Check signal propagation if specify
        if signal_assert:
            self.assertTrue((base_pattern.Iw.toarray()[:, 0] == self.got_matrices['Iw'].toarray()[:, 0]).all())
            self.assertTrue((base_pattern.propagate(self.X_base)[:, self.index_outputs].toarray() == self.y_base).all())

    def assertion_transient(self, transient_pattern, check_backward=False):

        # Check structure of dimension
        self.assertEqual(transient_pattern.depth, 3)
        self.assertEqual(transient_pattern.I.shape, (self.n_inputs, 3))
        self.assertEqual(transient_pattern.C.shape, (3, 3))
        self.assertEqual(transient_pattern.O.shape, (3, self.n_outputs))
        self.assertEqual(transient_pattern.I.nnz, sum([len(inds) for inds in self.transient_index]))
        self.assertEqual(len(transient_pattern.levels), 3)
        self.assertTrue((transient_pattern.levels == np.array([1, 1, 1])).all())
        self.assertEqual(transient_pattern.O.nnz, 1)

        # Check input link
        self.assertTrue((transient_pattern.Iw.toarray()[:, 0] == self.got_matrices['Iw'].toarray()[:, 1]).all())
        self.assertTrue((transient_pattern.Iw.toarray()[:, 1] == self.got_matrices['Iw'].toarray()[:, 2]).all())
        self.assertTrue((transient_pattern.Cw.toarray() == self.got_matrices['Cw'].toarray()[1:4, 1:4]).all())

        # Check signal propagation if specify
        self.assertTrue(
            (transient_pattern.propagate(self.X_transient)[:, self.index_outputs].toarray() == self.y_transient).all()
        )

        if not check_backward:
            return

        # Check backward values
        for i, l_inds in enumerate(self.transient_index):
            ax_firing = self.firing_graph.backward_firing['i'][l_inds, i + self.n_rand + 1].toarray()[:, 0]
            self.assertTrue((ax_firing == self.ax_backward_transient[i, :]).all())


def create_random_matrices(n_inputs, n_outputs, n_core, depth=2):
    """
    Create random matrices and levels, that can correspong to the structure of a firing graph with n_core vertices at
    layer 1, randomly linked to the input grid of bits. if depth is higher that 2, then then a single vertex is placed
    at each higher layer until desired depth is reached a the activation of any vertex of layer 1 result in the
    activation of last layer vertex.

    :param n_inputs: Number of bit of input grid.
    :param n_outputs: Number of output bit.
    :param n_core: Number of vertex of layer 1;
    :param depth: depth of the corresponding firing graph.
    :return: Sparse matrices of firing grah and an array of levels of its core vertex.
    """

    matrices = create_empty_matrices(n_inputs, n_outputs, n_core + depth - 2)

    matrices['Iw'] = csc_matrix(np.hstack((
        np.random.binomial(1, 0.5, (n_inputs,  n_core)),
        np.zeros((n_inputs, depth - 2))
    )))

    matrices['Cw'] = csc_matrix(np.vstack((
        np.hstack((
            np.zeros(2),
            np.ones(min(max(depth - 2, 0), 1)),
            np.zeros(max(depth - 3, 0))
        ))[np.newaxis, :].repeat(n_core, axis=0),
        np.eye(n_core + depth - 2)[n_core + 1:, :],
        np.zeros(n_core + depth - 2)
    )))

    matrices['Ow'] = csc_matrix(np.vstack((
        np.zeros((n_core + depth - 3, n_outputs)),
        np.random.binomial(1, 0.5, (1, n_outputs))
    )))

    ax_levels = np.hstack((np.random.randint(1, int(0.5 * n_inputs), n_core), np.ones(depth - 2)))

    return matrices, ax_levels


