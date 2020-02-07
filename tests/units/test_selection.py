# Global import
from firing_graph.core.data_structure.utils import create_empty_matrices
from scipy.sparse import csc_matrix
import numpy as np
import unittest

# Local import
from src.model.patterns import YalaBasePattern, YalaTransientPattern
from src.model.utils import get_transient_scores, overlap_test


class TestSelection(unittest.TestCase):

    def setUp(self):

        self.n_inputs, self.n_outputs, self.index_output, self.n_firing = 10, 1, 0, 100
        self.transient_index, self.precision_treshold = [[6, 7, 8, 9], [4, 5], [1, 2, 3], [0]], 0.5
        self.depth = 3

        self.transient_pattern = YalaTransientPattern.from_input_indices(
            self.n_inputs, self.n_outputs, 1, self.index_output, self.transient_index, 1
        )

        # Set drainer patterns
        self.drainer_params = {'weight': 10, 'p': 1, 'q': 1}

        # Set scores for test
        self.l_scores = [[30, 40, 70, 80], [50, 60], [1, 1, 1], [90]]

        # Create test transient pattern
        self.matrices = create_empty_matrices(self.n_inputs, self.n_outputs, 6)
        for i, l_inds in enumerate(self.transient_index):
            self.transient_pattern.matrices['Iw'][l_inds, i] = self.l_scores[i]
            self.transient_pattern.backward_firing['i'][l_inds, i] = [self.n_firing] * len(l_inds)

        # Set backward firing of last intersection lower than min_firing
        self.transient_pattern.backward_firing['i'][
            self.transient_index[-1], self.transient_pattern.n_intersection - 1
        ] = self.n_firing - 1

        # Build expected result
        self.scored_bits = [
            (0, {"index": [9], "precision": None}),
            (0, {"index": [8], "precision": None}),
            (1, {"index": [5], "precision": None}),
            (1, {"index": [4], "precision": None}),
            (0, {"index": [7], "precision": None}),
            (0, {"index": [6], "precision": None}),
        ]

        # Build variable to test overlap
        self.overlap_rate = 0.667
        self.ax_mask = np.array([True, True, False, True])
        self.ax_base = np.array([True, True, False, False])
        self.test_overlap_pattern = YalaBasePattern.from_partition(
            {"indices": [self.transient_pattern.n_intersection - 1]}, self.transient_pattern, self.index_output
        )

        self.sax_input_1 = csc_matrix([
            [True] + [False] * 9,
            [True] + [False] * 9,
            [False] + [False] * 9,
            [False] + [False] * 9
            ])

        self.sax_input_2 = csc_matrix([
            [True] + [False] * 9,
            [True] + [False] * 9,
            [False] + [False] * 9,
            [True] + [False] * 9
            ])

    def test_scoring_bits(self):
        """
        python -m unittest tests.units.test_selection.TestSelection.test_scoring_bits

        """
        # Test base pattern from indices
        l_scores = get_transient_scores(
            self.transient_pattern, self.n_firing, self.drainer_params, precision_treshold=self.precision_treshold
        )

        # Check results
        for i, t in enumerate(l_scores):
            self.assertEqual(t[0], self.scored_bits[i][0])
            self.assertEqual(t[1]['index'], self.scored_bits[i][1]['index'])

    def test_overlap(self):
        """
        python -m unittest tests.units.test_selection.TestSelection.test_overlap

        """
        # test overlap case 1
        _, do_overlap = overlap_test(
            self.sax_input_1, self.ax_base, self.ax_mask, self.test_overlap_pattern, self.overlap_rate
        )
        self.assertTrue(do_overlap)

        # test overlap case 2
        _, do_overlap = overlap_test(
            self.sax_input_2, self.ax_base, self.ax_mask, self.test_overlap_pattern, self.overlap_rate
        )
        self.assertTrue(not do_overlap)
