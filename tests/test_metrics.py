import unittest

from common import get_precision_at_k, get_average_precision, get_mean_rank, get_mean_reciprocal_rank, get_mean_average_precision

class TestMetrics(unittest.TestCase):

    def test_precision_at_k(self):
        result_prob = [(0, 1, 0.1), (0, 2, 0.3), (1, 2, 0.5), (1, 4, 0.2), (2, 4, 0.9)]
        true_pairs = [(0, 1), (2,4)]
        self.assertEqual(get_precision_at_k(result_prob, true_pairs, k=1), 1)
        self.assertEqual(get_precision_at_k(result_prob, true_pairs, k=2), 0.5)
        self.assertEqual(get_precision_at_k(result_prob, true_pairs, k=5), 0.4)
        self.assertEqual(get_precision_at_k(result_prob, [], k=5), 0)

        result_prob = [(0 , 1, 0.9), (1, 2, 0.4), (2, 3, 0.5)]
        true_pairs = [(0,1)]
        self.assertEqual(get_precision_at_k(result_prob, true_pairs, k=1), 0)
        self.assertEqual(get_precision_at_k(result_prob, true_pairs, k=2), 0)
        self.assertEqual(round(get_precision_at_k(result_prob, true_pairs, k=3),2), 0.33)


    def test_get_average_precision(self):
        result_prob = [(0, 1, 0.1), (0, 2, 0.3), (1, 2, 0.5), (1, 4, 0.2), (2, 4, 0.9)]
        true_pairs = [(0, 1), (2,4)]
        self.assertEqual(get_average_precision(result_prob, true_pairs), 0.7)
        self.assertEqual(get_average_precision(result_prob[:4], true_pairs[:1]), 1)
        self.assertEqual(get_average_precision(result_prob, []), 0)

        result_prob = [(0, 1, 0.1), (1, 4, 0.2), (0, 2, 0.3), (1, 2, 0.5), (2, 3, 0.6), (2, 4, 0.9),
                        (4, 5, 1.0), (4, 6, 1.1), (4, 7, 1.15), (5, 6, 1.3)]
        true_pairs = [(0, 1), (1, 4), (2, 4), (1, 2), (5, 6)]
        self.assertEqual(round(get_average_precision(result_prob, true_pairs), 2), 0.78)


    def test_mean_rank(self):
        result_prob = [(0, 1, 0.1), (0, 2, 0.3), (1, 2, 0.5), (2, 3, 0.2), (2, 4, 0.9)]
        true_pairs = [(0, 1), (2,4)]

        self.assertEqual(get_mean_rank(result_prob,true_pairs), 1.5)
        self.assertEqual(get_mean_rank(result_prob[:4],true_pairs[:1]), 1)
        self.assertEqual(get_mean_rank(result_prob, []), 0)

        result_prob = [(0, 2, 0.1), (0, 1, 0.2), (2, 3, 0.1), (2, 4, 0.5), (3, 1, 0.2), (3, 2, 0.4), (3, 4, 0.8)]
        true_pairs = [(0, 1), (2, 3), (3, 4)]
        self.assertEqual(get_mean_rank(result_prob,true_pairs), 2)

    def test_mean_reciprocal_rank(self):
        result_prob = [(0, 1, 0.1), (0, 2, 0.3), (1, 2, 0.5), (2, 3, 0.2), (2, 4, 0.9)]
        true_pairs = [(0, 1), (2,4)]

        self.assertEqual(get_mean_reciprocal_rank(result_prob,true_pairs), 0.75)
        self.assertEqual(get_mean_reciprocal_rank(result_prob[:4],true_pairs[:1]), 1)
        self.assertEqual(get_mean_reciprocal_rank(result_prob, []), 0)

        result_prob = [(0, 2, 0.1), (0, 1, 0.2), (2, 3, 0.1), (2, 4, 0.5), (3, 1, 0.2), (3, 2, 0.4), (3, 4, 0.8)]
        true_pairs = [(0, 1), (2, 3), (3, 4)]
        self.assertEqual(round(get_mean_reciprocal_rank(result_prob, true_pairs), 2), 0.61)

        result_prob = [(0, 1, 0.1), (0, 2, 0.2), (0, 3, 0.3), (0, 4, 0.4), (0, 5, 0.5), (0, 6, 0.6),
                        (1, 0, 0.1), (1, 2, 0.2),(1, 3, 0.3), (1, 4, 0.4), (1, 5, 0.5), (1, 6, 0.6),]
        true_pairs = [(0, 1), (0, 4), (0, 5), (0, 6), (1, 4), (1, 5), (1, 6)]
        self.assertEqual(round(get_mean_reciprocal_rank(result_prob, true_pairs), 2), 0.63)


    def test_mean_average_precision(self):
        result_prob = [(0, 1, 0.1), (0, 2, 0.3), (1, 2, 0.5), (2, 3, 0.2), (2, 4, 0.9)]
        true_pairs = [(0, 1), (2,4)]

        self.assertEqual(get_mean_average_precision(result_prob, true_pairs), 0.75)
        self.assertEqual(get_mean_average_precision(result_prob[:4],true_pairs[:1]), 1)
        self.assertEqual(get_mean_average_precision(result_prob, []), 0)

        result_prob = [(0, 1, 0.1), (0, 2, 0.2), (0, 3, 0.3), (0, 4, 0.4), (0, 5, 0.5), (0, 6, 0.6),
                        (1, 0, 0.1), (1, 2, 0.2),(1, 3, 0.3), (1, 4, 0.4), (1, 5, 0.5), (1, 6, 0.6),]
        true_pairs = [(0, 1), (0, 4), (0, 5), (0, 6), (1, 4), (1, 5), (1, 6)]
        self.assertEqual(round(get_mean_average_precision(result_prob, true_pairs), 2), 0.54)

        result_prob = [(0, 2, 0.1), (0, 1, 0.2), (2, 3, 0.1), (2, 4, 0.5), (3, 1, 0.2), (3, 2, 0.4), (3, 4, 0.8)]
        true_pairs = [(0, 1), (2, 3), (3, 4)]
        self.assertEqual(round(get_mean_average_precision(result_prob, true_pairs), 2), 0.61)