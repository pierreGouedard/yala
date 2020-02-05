# Global import
from firing_graph.core.data_structure.graph import FiringGraph

from scipy.sparse import csc_matrix, lil_matrix, hstack, vstack
import numpy as np
import random
import unittest

# Local import
from src.model.yala import Yala


class TestSelection(unittest.TestCase):

    def setUp(self):

        # TODO: This test module should be used to test function in src.model.utils, especially
        #  * extract_draining_pattern
        #  * get_transient_scores
        #  * get_bit_scores
        #  * get_precision

        raise NotImplementedError

    def test_base_pattern(self):
        """
        python -m unittest tests.units.test_selection.TestSelection.test_base_pattern

        """

        # Test base pattern from indices

        # Test base pattern from partition of a larger firing graph

        # test augment base pattern

        raise NotImplementedError

    def test_transient_pattern(self):
        """
        python -m unittest tests.units.test_selection.TesPattern.test_transient_pattern

        """

        # test transient pattern from indices

        # test transient pattern from a larger graph

        raise NotImplementedError

    def test_draining_pattern(self):

        # Test creation of single draining pattern

        # test creation of a multiple draining pattern

        raise NotImplementedError

    def test_predicting_pattern(self):

        # Test creation of single draining pattern

        # test creation of a multiple draining pattern

        raise NotImplementedError
