# Global import
from firing_graph.core.data_structure.graph import FiringGraph

from scipy.sparse import csc_matrix, lil_matrix, hstack, vstack
import numpy as np
import random
import unittest

# Local import
from src.model.yala import Yala


class TesStructure(unittest.TestCase):

    def setUp(self):

        # Set firing graph to create base structure
        self.n_inputs, self.n_outputs, self.n_vertices = 100, 10, 2
        self.n_core_bs, self.depth_bs = 5, 4
        self.drainer_params, self.expected_precision = {'t': 100, 'weight': 1, 'p': 1, 'q': 1}, 0.5
        self.ax_C_bs = np.array([
            [False, False, True, True, False],
            [False, False, False, True, False],
            [False, False, False, False, True],
            [False, False, False, False, True],
            [False, False, False, False, False],
        ])

        # Init base structure
        sax_I = lil_matrix((self.n_inputs, (self.n_vertices + 1) * self.n_outputs))
        sax_C = lil_matrix(((self.n_vertices + 1) * self.n_outputs, (self.n_vertices + 1) * self.n_outputs))
        sax_O = lil_matrix(((self.n_vertices + 1) * self.n_outputs, self.n_outputs))

        # Create base params
        d_masks = {"Im": lil_matrix(sax_I.shape), "Cm": lil_matrix(sax_C.shape), "Om": lil_matrix(sax_O.shape)}
        ax_levels, sax_score = np.ones((self.n_vertices + 1) * self.n_outputs), lil_matrix(sax_I.shape)

        # Build partitioned firing graph in case of
        self.firing_graph1 = create_partition_base(
            self.n_inputs, self.n_outputs, self.n_vertices + 1, sax_I, sax_O, sax_C, sax_score, ax_levels, d_masks
        )

        # Init base structure
        sax_I = lil_matrix((self.n_inputs, (self.n_vertices + 8) * self.n_outputs))
        sax_C = lil_matrix(((self.n_vertices + 8) * self.n_outputs, (self.n_vertices + 8) * self.n_outputs))
        sax_O = lil_matrix(((self.n_vertices + 8) * self.n_outputs, self.n_outputs))

        # Create base params
        d_masks = {"Im": lil_matrix(sax_I.shape), "Cm": lil_matrix(sax_C.shape), "Om": lil_matrix(sax_O.shape)}
        ax_levels, sax_score = np.ones((self.n_vertices + 8) * self.n_outputs), lil_matrix(sax_I.shape)
        self.l_base_indices = [[0, 1], [2, 3]]

        self.firing_graph2 = create_partition_augment(
            self.n_inputs, self.n_outputs, self.n_vertices + 8, sax_I, sax_O, sax_C, sax_score, ax_levels, d_masks,
            self.l_base_indices
        )

    def test_create_base_structures(self):
        """
        python -m unittest tests.units.test_structure_operations.TesStructure.test_create_base_structures

        """
        yala = Yala()
        yala.drainer_params.update(self.drainer_params)

        for partition in self.firing_graph1.partitions:

            # Randomly flip output
            is_output_flipped = random.random() < 0.5

            l_structures, changed = yala.create_base_structures(
                partition, self.firing_graph1, is_output_flipped=is_output_flipped
            )

            # Select random structure to make test on
            (ind, structure) = random.choice(list(enumerate(l_structures)))

            # Test general properties of  whats returned
            self.assertTrue(changed)
            self.assertEqual(len(l_structures), self.n_vertices)

            # Check structure shapes
            self.assertEqual(structure.I.shape[0], self.n_inputs)
            self.assertEqual(structure.C.shape[0], 5)
            self.assertEqual(structure.O.shape[1], self.n_outputs)
            self.assertTrue((structure.C.toarray() == self.ax_C_bs).all())

            # Make sure best bits has been selected
            self.assertTrue(structure.I[partition['indices_inputs'][ind][0], int(is_output_flipped)] > 0)
            self.assertTrue(structure.I[partition['indices_inputs'][ind][1], int(is_output_flipped)] == 0)
            self.assertTrue(structure.I[partition['indices_inputs'][ind][2], int(is_output_flipped)] == 0)

            # Make sure precision is well computed
            self.assertAlmostEqual(structure.precision, self.expected_precision, delta=1e-2)

    def test_update_base_structures(self):
        """
        python -m unittest tests.units.test_structure_operations.TesStructure.test_update_base_structures

        """

        yala = Yala()
        yala.drainer_params.update(self.drainer_params)

        for partition in self.firing_graph2.partitions:

            # Randomly flip output
            is_output_flipped = random.random() < 0.5

            l_structures, changed = yala.update_base_structures(
                partition, self.firing_graph2, is_output_flipped=is_output_flipped
            )

            # Select random structure to make test on
            (ind, structure) = random.choice(list(enumerate(l_structures)))

            # Test general properties of  whats returned
            self.assertTrue(changed)
            self.assertEqual(len(l_structures), self.n_vertices)

            # Check structure shapes
            self.assertEqual(structure.I.shape[0], self.n_inputs)
            self.assertEqual(structure.C.shape[0], 5)
            self.assertEqual(structure.O.shape[1], self.n_outputs)
            self.assertTrue((structure.C.toarray() == self.ax_C_bs).all())

            # Make sure best bits has been selected from transient
            self.assertTrue(structure.I[partition['indices_inputs_transient'][ind][0], int(is_output_flipped)] > 0)
            self.assertTrue(structure.I[partition['indices_inputs_transient'][ind][1], int(is_output_flipped)] == 0)
            self.assertTrue(structure.I[partition['indices_inputs_transient'][ind][2], int(is_output_flipped)] == 0)

            # Make sure base bits still be linked to base structure
            self.assertTrue((structure.I[self.l_base_indices[0], 0].toarray() > 0).all())
            self.assertTrue((structure.I[self.l_base_indices[1], 1].toarray() > 0).all())


def create_partition_base(n_inputs, n_outputs, n_vertices, sax_I, sax_O, sax_C, sax_score, ax_levels, d_masks):

    # Build partitions fro create
    partitions, n_core = [], 0
    for i in range(n_outputs):

        # Make partition, augmented with ground of truth
        partitions.append({
            'indices': [n_core + i for i in range(n_vertices)],
            'indices_inputs': [],
            'depth': 3
        })

        n_core += n_vertices

        # Select inputs
        for n, j in enumerate(partitions[-1]['indices'][:-1]):

            # Select randomly input indices
            step = int(n_inputs / (n_vertices - 1))
            l_input_indices = np.random.choice(range(n * step, (n + 1) * step), size=3, replace=False)

            # Update structure
            sax_I[l_input_indices, j] = 1
            sax_I[l_input_indices[0], j] += 1
            sax_C[j, partitions[-1]['indices'][-1]] = 1

            # Add ground of truth to partitions
            partitions[-1]['indices_inputs'].append(l_input_indices)

        # Update Output matrix
        sax_O[partitions[-1]['indices'][-1], i] = 1

    firing_graph = FiringGraph.from_matrices(
        sax_I.tocsc(), sax_C.tocsc(), sax_O.tocsc(), ax_levels, mask_matrices=d_masks, depth=3,
        partitions=partitions
    )

    return firing_graph


def create_partition_augment(n_inputs, n_outputs, n_vertices, sax_I, sax_O, sax_C, sax_score, ax_levels, d_masks,
                             l_base_indices):

    # Build partitions fro create
    partitions, n_core = [], 0
    for i in range(n_outputs):

        # Make partition, augmented with ground of truth
        partitions.append({
            'indices': [n_core + i for i in range(n_vertices)],
            'indices_inputs_transient': [],
            'depth': 5,
            'precision': 0,
            'partitions': [
                {'indices': [i for i in range(5)], 'name': 'base', 'depth': 4},
                {'indices': [5 + i for i in range(n_vertices - 7)], 'name': 'transient', 'depth': 3},
            ]
        })

        n_core += n_vertices

        # Set base structure
        sax_I[l_base_indices[0], partitions[-1]['indices'][0]] = 1
        sax_I[l_base_indices[1], partitions[-1]['indices'][1]] = 1
        sax_C[partitions[-1]['indices'][0], partitions[-1]['indices'][2]] = 1
        sax_C[partitions[-1]['indices'][0], partitions[-1]['indices'][3]] = 1
        sax_C[partitions[-1]['indices'][1], partitions[-1]['indices'][3]] = 1
        sax_C[partitions[-1]['indices'][2], partitions[-1]['indices'][4]] = 1
        sax_C[partitions[-1]['indices'][3], partitions[-1]['indices'][4]] = 1
        sax_O[partitions[-1]['indices'][4], i] = 1

        # Select inputs for transient structure
        for n, j in enumerate(partitions[-1]['indices'][5:-3]):

            # Select randomly input indices
            step = int((n_inputs - 4) / (n_vertices - 8))
            l_input_indices = np.random.choice(range(4 + n * step, (n + 1) * step), size=3, replace=False)

            # Update structure
            sax_I[l_input_indices, j] = 1
            sax_I[l_input_indices[0], j] += 1
            sax_C[j, partitions[-1]['indices'][-3]] = 1

            # Add ground of truth to partitions
            partitions[-1]['indices_inputs_transient'].append(l_input_indices)

        # Update Output matrix
        sax_O[partitions[-1]['indices'][-1], i] = 1
        sax_C[partitions[-1]['indices'][-3], partitions[-1]['indices'][-2]] = 1
        sax_C[partitions[-1]['indices'][-2], partitions[-1]['indices'][-1]] = 1
        sax_C[partitions[-1]['indices'][4], partitions[-1]['indices'][-1]] = 1

    firing_graph = FiringGraph.from_matrices(
        sax_I.tocsc(), sax_C.tocsc(), sax_O.tocsc(), ax_levels, mask_matrices=d_masks, depth=5,
        partitions=partitions
    )

    return firing_graph