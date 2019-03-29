import copy
import pandas as pd
import unittest

from common import (
    export_embeddings,
    export_result_prob,
    get_optimal_threshold,
    get_logger,
    InformationRetrievalMetrics,
    log_quality_results,
    sigmoid)
from data.cora import Cora
from data.febrl import FEBRL
from data.census import Census
from ERER.model import Graph_ERER
from ERER.mtranse import MTransE
from scipy import spatial

class TestMTransE(unittest.TestCase):

    def _test_mtranse(self, dataset, params):
        model = dataset()
        graph = Graph_ERER(dataset)
        logger = get_logger('RL.Test.MTransE.' + str(model))

        mtranse = MTransE(graph, dimension=params['dimension'],
                                 batchSize=params['batchSize'],
                                 learning_rate=params['learning_rate'],
                                 regularizer_scale = params['regularizer_scale'],
                                 alpha=params['alpha'],
                                 neg_rate=params['neg_rate'],
                                 neg_rel_rate=params['neg_rel_rate'],
                                 margin=params['margin'])
        mtranse.train(max_epochs=params['max_epochs'])

        ent_embeddings_a = mtranse.get_ent_embeddings_A()
        ent_embeddings_b = mtranse.get_ent_embeddings_B()
        result_prob = []
        for i in range(0, len(graph.entity_pairs)):
            distance = abs(spatial.distance.cosine(
                                ent_embeddings_a[int(graph.entity_pairs[i][0])],
                                ent_embeddings_b[int(graph.entity_pairs[i][1])]))
            result_prob.append((graph.entity_pairs[i][0], graph.entity_pairs[i][1], distance))
            #logger.info("i: %d, distance: %f true_pairs: %s", i, distance, graph.entity_pairs[i] in true_pairs)

        #Write Embeddings to file
        export_embeddings('erer', str(model), 'MTransE', graph.entityA, ent_embeddings_a)
        export_embeddings('erer', str(model), 'MTransE', graph.entityB, ent_embeddings_b)
        export_result_prob(dataset, 'erer', str(model), 'MTransE', graph.entityA, result_prob, graph.true_pairs, graph.entityB)
        optimal_threshold, max_fscore = get_optimal_threshold(result_prob, graph.true_pairs)

        try:
            params['threshold'] = optimal_threshold
            result = pd.MultiIndex.from_tuples([(e1, e2) for (e1, e2, d) in result_prob if d <= optimal_threshold])
            log_quality_results(logger, result, graph.true_pairs, len(graph.entity_pairs), params)
        except Exception as e:
            logger.info("Zero Reults")
            logger.error(e)

        #Log MAP, MRR and Hits@K
        ir_metrics = InformationRetrievalMetrics(result_prob, graph.true_pairs)
        prec_at_1 = ir_metrics.log_metrics(logger, params)

        mtranse.close_tf_session()
        return (max_fscore, prec_at_1)

    def get_default_params(self):
        return {'learning_rate': 0.1, 'dimension': 128, 'max_epochs': 100, 'alpha' : 5,
                'regularizer_scale' : 0.1, 'batchSize' : 512, 'margin' : 2, 'neg_rate' : 7,
                'neg_rel_rate' : 1 }

    def test_cora(self):
        self._test_mtranse(Cora, self.get_default_params())

    def test_febrl(self):
        self._test_mtranse(FEBRL, self.get_default_params())

    def test_census(self):
        self._test_mtranse(Census, self.get_default_params())

    def _test_grid_search(self, dataset):
        learning_rate = [0.1]
        dimension = [128, 256]
        max_epochs = [1000]
        alpha = [5, 10]
        regularizer_scale = [0.1]
        batchSize = [1024, 512]
        margin = [1, 2]
        neg_rate = [7, 4]
        neg_rel_rate = [1]

        max_fscore = 0
        max_prec_at_1 = 0
        count = 0

        for lr, d, me, a, reg, bs, m, nr, nrr in itertools.product(learning_rate, dimension, \
                max_epochs, alpha, regularizer_scale, batchSize, margin, neg_rate, neg_rel_rate):

            params = {'learning_rate': lr, 'dimension': d, 'max_epochs': me, 'alpha' : a,
                'regularizer_scale' : reg, 'batchSize' : bs, 'margin' : m, 'neg_rate' : nr,
                'neg_rel_rate' : nrr}
            cur_fscore, cur_prec_at_1 = self._test_mtranse(dataset, params)
            count = count + 1
            if max_fscore <= cur_fscore:
                max_fscore = cur_fscore
            if max_prec_at_1 <= cur_prec_at_1:
                max_prec_at_1 = cur_prec_at_1

            logger.info("Ran total %d Tests.", count)
            logger.info("Max Fscore: %f", max_fscore)
            logger.info("Max Precision@1: %f", max_prec_at_1)

    def test_grid_cora(self):
        self._test_grid_search(Cora)

    def test_grid_febrl(self):
        self._test_grid_search(FEBRl)

    def test_grid_census(self):
        self._test_grid_search(Census)