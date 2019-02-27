import unittest

from common import InformationRetrievalMetrics


class TestMetrics(unittest.TestCase):

    def test_mean_precision_at_k(self):
        result_prob = [(0, 1, 0.1), (0, 2, 0.3), (1, 2, 0.5), (1, 4, 0.2), (2, 4, 0.9), (2, 3, 1)]
        true_pairs = [(0, 1), (2,4)]
        ir_metrics = InformationRetrievalMetrics(result_prob, true_pairs)
        self.assertEqual(ir_metrics.get_mean_precisison_at_k(k=1), 1)
        self.assertEqual(ir_metrics.get_mean_precisison_at_k(k=2), 0.5)

        result_prob = [(0 , 1, 0.9), (1, 2, 0.4), (2, 3, 0.5), (0, 2, 0.2), (0, 3, 0.5)]
        true_pairs = [(0, 1)]
        ir_metrics = InformationRetrievalMetrics(result_prob, true_pairs)
        self.assertEqual(ir_metrics.get_mean_precisison_at_k(k=1), 0)
        self.assertEqual(ir_metrics.get_mean_precisison_at_k(k=2), 0)
        self.assertEqual(round(ir_metrics.get_mean_precisison_at_k(k=3), 2), 0.33)

    def test_mean_reciprocal_rank(self):
        result_prob = [(0, 1, 0.1), (0, 2, 0.3), (1, 2, 0.5), (2, 3, 0.2), (2, 4, 0.9)]
        true_pairs = [(0, 1), (2,4)]

        ir_metrics = InformationRetrievalMetrics(result_prob, true_pairs)
        self.assertEqual(ir_metrics.get_mean_reciprocal_rank(), 0.75)

        ir_metrics = InformationRetrievalMetrics(result_prob[:4], true_pairs[:1])
        self.assertEqual(ir_metrics.get_mean_reciprocal_rank(), 1)

        result_prob = [(0, 2, 0.1), (0, 1, 0.2), (2, 3, 0.1), (2, 4, 0.5), (3, 1, 0.2), (3, 2, 0.4), (3, 4, 0.8)]
        true_pairs = [(0, 1), (2, 3), (3, 4)]
        ir_metrics = InformationRetrievalMetrics(result_prob, true_pairs)
        self.assertEqual(round(ir_metrics.get_mean_reciprocal_rank(), 2), 0.61)

        result_prob = [(0, 1, 0.1), (0, 2, 0.2), (0, 3, 0.3), (0, 4, 0.4), (0, 5, 0.5), (0, 6, 0.6),
                        (1, 0, 0.1), (1, 2, 0.2),(1, 3, 0.3), (1, 4, 0.4), (1, 5, 0.5), (1, 6, 0.6),]
        true_pairs = [(0, 1), (0, 4), (0, 5), (0, 6), (1, 4), (1, 5), (1, 6)]
        ir_metrics = InformationRetrievalMetrics(result_prob, true_pairs)
        self.assertEqual(round(ir_metrics.get_mean_reciprocal_rank(), 2), 0.63)

    def test_mean_average_precision(self):
        result_prob = [(0, 1, 0.1), (0, 2, 0.3), (1, 2, 0.5), (2, 3, 0.2), (2, 4, 0.9)]
        true_pairs = [(0, 1), (2,4)]

        ir_metrics = InformationRetrievalMetrics(result_prob, true_pairs)
        self.assertEqual(ir_metrics.get_mean_average_precision(), 0.75)

        ir_metrics = InformationRetrievalMetrics(result_prob[:4], true_pairs[:1])
        self.assertEqual(ir_metrics.get_mean_average_precision(), 1)

        result_prob = [(0, 1, 0.1), (0, 2, 0.2), (0, 3, 0.3), (0, 4, 0.4), (0, 5, 0.5), (0, 6, 0.6),
                        (1, 0, 0.1), (1, 2, 0.2),(1, 3, 0.3), (1, 4, 0.4), (1, 5, 0.5), (1, 6, 0.6),]
        true_pairs = [(0, 1), (0, 4), (0, 5), (0, 6), (1, 4), (1, 5), (1, 6)]
        ir_metrics = InformationRetrievalMetrics(result_prob, true_pairs)
        self.assertEqual(round(ir_metrics.get_mean_average_precision(), 2), 0.54)

        result_prob = [(0, 2, 0.1), (0, 1, 0.2), (2, 3, 0.1), (2, 4, 0.5), (3, 1, 0.2), (3, 2, 0.4), (3, 4, 0.8)]
        true_pairs = [(0, 1), (2, 3), (3, 4)]
        ir_metrics = InformationRetrievalMetrics(result_prob, true_pairs)
        self.assertEqual(round(ir_metrics.get_mean_average_precision(), 2), 0.61)