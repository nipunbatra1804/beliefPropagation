import unittest

from src.tree import Tree
from numpy import testing
import numpy as np





class TestTree(unittest.TestCase):
    def assertNp(self, assertFunc):
        self.assertIsNone(assertFunc)

    def test_numpy_assert_array_equal(self):
        self.assertNp(testing.assert_array_equal(np.array([1, 2]), np.array([1, 2])))

    def test_setup_self_factor(self):
        




if __name__ == '__main__':
    unittest.main()
