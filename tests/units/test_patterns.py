# Global import
from firing_graph.core.data_structure.graph import FiringGraph

from scipy.sparse import csc_matrix, lil_matrix, hstack, vstack
import numpy as np
import random
import unittest

# Local import
from src.model.yala import Yala


class TestPattern(unittest.TestCase):

    def setUp(self):

        # TODO: for all test we should test
        #  * matrices dimension
        #  * signal propagation

        raise NotImplementedError

    def test_base_pattern(self):
        """
        python -m unittest tests.units.test_patterns.TestPattern.test_base_pattern

        """

        # Test base pattern from indices

        # Test base pattern from partition of a larger firing graph

        # test augment base pattern

        raise NotImplementedError

    def test_transient_pattern(self):
        """
        python -m unittest tests.units.test_patterns.TestPattern.test_transient_pattern

        """

        # test transient pattern from indices

        # test transient pattern from a larger graph

        raise NotImplementedError

    def test_draining_pattern(self):
        """
        python -m unittest tests.units.test_patterns.TestPattern.test_transient_pattern

        """
        # Test creation of single draining pattern

        # test creation of a multiple draining pattern

        raise NotImplementedError

    def test_predicting_pattern(self):
        """
        python -m unittest tests.units.test_patterns.TestPattern.test_transient_pattern

        """
        # Test creation of single draining pattern

        # test creation of a multiple draining pattern

        raise NotImplementedError
