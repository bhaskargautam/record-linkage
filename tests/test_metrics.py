import unittest

from common import get_precision_at_k, get_average_precision, get_mean_rank, get_mean_reciprocal_rank, get_mean_average_precision_at_k

class TestMetrics(unittest.TestCase):

    def test_precision_at_k(self):
        result_prob = [(0, 1, 0.1), (0, 2, 0.3), (1, 2, 0.5), (1, 4, 0.2), (2, 4, 0.9)]
        true_pairs = [(0, 1), (2,4)]
        self.assertEqual(get_precision_at_k(result_prob, true_pairs, k=1), 1)
        self.assertEqual(get_precision_at_k(result_prob, true_pairs, k=2), 0.5)
        self.assertEqual(get_precision_at_k(result_prob, true_pairs, k=5), 0.4)
        self.assertEqual(get_precision_at_k(result_prob, [], k=5), 0)


    def test_get_average_precision(self):
        result_prob = [(0, 1, 0.1), (0, 2, 0.3), (1, 2, 0.5), (1, 4, 0.2), (2, 4, 0.9)]
        true_pairs = [(0, 1), (2,4)]
        self.assertEqual(get_average_precision(result_prob, true_pairs), 0.7)
        self.assertEqual(get_average_precision(result_prob[:4], true_pairs[:1]), 1)
        self.assertEqual(get_average_precision(result_prob, []), 0)

    def test_mean_rank(self):
        result_prob = [(0, 1, 0.1), (0, 2, 0.3), (1, 2, 0.5), (2, 3, 0.2), (2, 4, 0.9)]
        true_pairs = [(0, 1), (2,4)]

        self.assertEqual(get_mean_rank(result_prob,true_pairs), 1.5)
        self.assertEqual(get_mean_rank(result_prob[:4],true_pairs[:1]), 1)
        self.assertEqual(get_mean_rank(result_prob, []), 0)

    def test_mean_reciprocal_rank(self):
        result_prob = [(0, 1, 0.1), (0, 2, 0.3), (1, 2, 0.5), (2, 3, 0.2), (2, 4, 0.9)]
        true_pairs = [(0, 1), (2,4)]

        self.assertEqual(get_mean_reciprocal_rank(result_prob,true_pairs), 0.5)
        self.assertEqual(get_mean_reciprocal_rank(result_prob[:4],true_pairs[:1]), 1)
        self.assertEqual(get_mean_reciprocal_rank(result_prob, []), 0)

    def test_mean_average_precision_at_k(self):
        result_prob = [(0, 1, 0.1), (0, 2, 0.3), (1, 2, 0.5), (2, 3, 0.2), (2, 4, 0.9)]
        true_pairs = [(0, 1), (2,4)]

        self.assertEqual(get_mean_average_precision_at_k(result_prob, true_pairs), 0.5)
        self.assertEqual(get_mean_average_precision_at_k(result_prob, true_pairs, k=2), 0.5)
        self.assertEqual(get_mean_average_precision_at_k(result_prob[:4],true_pairs[:1]), 1)
        self.assertEqual(get_mean_average_precision_at_k(result_prob, []), 0)
