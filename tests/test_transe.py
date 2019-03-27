import config
import itertools
import pandas as pd
import numpy as np
import recordlinkage
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
from ER.model import Graph_ER
from ER.transe import TransE
from scipy import spatial
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import precision_recall_curve

class TestTransE(unittest.TestCase):
    def _test_transe(self, dataset, params):
        #Load Graph Data
        graph = Graph_ER(dataset)
        model = dataset()
        logger = get_logger('RL.Test.TransE.' + str(model))

        transe = TransE(graph, dimension=params['dimension'],
                        learning_rate=params['learning_rate'],
                        margin=params['margin'],
                        regularizer_scale=params['regularizer_scale'],
                        batchSize=params['batchSize'],
                        neg_rate=params['neg_rate'],
                        neg_rel_rate=params['neg_rel_rate'])
        loss = transe.train(max_epochs=params['epochs'])
        logger.info("Training Complete with loss: %f", loss)

        ent_embeddings = transe.get_ent_embeddings()

        result_prob = []
        for i in range(0, len(graph.entity_pairs)):
            distance = abs(spatial.distance.cosine(
                                ent_embeddings[graph.entity_pairs[i][0]],
                                ent_embeddings[graph.entity_pairs[i][1]]))
            result_prob.append((graph.entity_pairs[i][0], graph.entity_pairs[i][1], distance))
            #logger.info("i: %d, distance: %f true_pairs: %s", i, distance, graph.entity_pairs[i] in true_pairs)

        #Write Embeddings to file
        export_embeddings('er', str(model), 'TransE', graph.entity, ent_embeddings)
        export_result_prob(dataset, 'er', str(model), 'TransE', graph.entity, result_prob, graph.true_pairs)
        optimal_threshold, max_fscore = get_optimal_threshold(result_prob, graph.true_pairs)

        try:
            params['threshold'] = optimal_threshold
            result = pd.MultiIndex.from_tuples([(e1, e2) for (e1, e2, d) in result_prob if d <= optimal_threshold])
            log_quality_results(logger, result, graph.true_pairs, len(graph.entity_pairs), params)
        except:
            logger.info("Zero Reults")

        #Log MAP, MRR and Hits@K
        ir_metrics = InformationRetrievalMetrics(result_prob, graph.true_pairs)
        precison_at_1 = ir_metrics.log_metrics(logger, params)

        transe.close_tf_session()
        return (max_fscore, precison_at_1)

    def get_default_params(self):
        return {'learning_rate': 0.1, 'margin': 2, 'dimension': 128, 'epochs': 1000,
                'regularizer_scale' : 0.1, 'batchSize' : 1024, 'neg_rate' : 8, 'neg_rel_rate': 2}

    def test_transe_cora(self):
        self._test_transe(Cora, self.get_default_params())

    def test_transe_febrl(self):
        self._test_transe(FEBRL, self.get_default_params())

    def test_transe_census(self):
        self._test_transe(Census, self.get_default_params())

    def _test_grid_search(self, dataset):
        dimension= [128, 256]
        batchSize= [1024, 32]
        learning_rate= [0.1]
        margin= [1]
        regularizer_scale = [0.1]
        epochs = [1000, 5000]
        neg_rel_rate = [7, 12]
        neg_rate = [1, 4]
        count = 0
        max_fscore = 0
        max_prec_at_1 = 0

        model = dataset()
        logger = get_logger('RL.Test.GridSearch.TransE.' + str(model))

        for d, bs, lr, m, reg, e, nr, nrr in \
            itertools.product(dimension, batchSize, learning_rate, margin, regularizer_scale, epochs, neg_rate, neg_rel_rate):
            params = {'learning_rate': lr, 'margin': m, 'dimension': d, 'epochs': e, 'batchSize' : bs,
                        'regularizer_scale' : reg, 'neg_rate' : nr, 'neg_rel_rate': nrr}
            logger.info("\nTest:%d, PARAMS: %s", count, str(params))
            count = count + 1

            cur_fscore, cur_prec_at_1 = self._test_transe(dataset, params)
            if max_fscore <= cur_fscore:
                max_fscore = cur_fscore
            if max_prec_at_1 <= cur_prec_at_1:
                max_prec_at_1 = cur_prec_at_1

            logger.info("Ran total %d Tests.", count)
            logger.info("Max Fscore: %f", max_fscore)
            logger.info("Max Mean Precision@1: %f", max_prec_at_1)

    def test_grid_search_cora(self):
        self._test_grid_search(Cora)

    def test_grid_search_febrl(self):
        self._test_grid_search(FEBRL)

    def test_grid_search_census(self):
        self._test_grid_search(Census)