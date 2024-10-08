
# Tests for the pyro backend
import unittest

class TestPyroBackend(unittest.TestCase):

    def test_sanity_check(self):
        # Sanity Check
        self.assertEqual(1, 1)


if __name__ == '__main__':
    unittest.main()