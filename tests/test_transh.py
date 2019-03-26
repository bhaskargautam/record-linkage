import config
import itertools
import pandas as pd
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
from ER.transh import TransH
from ER.model import Graph_ER
from scipy import spatial

class TestTransH(unittest.TestCase):

    def _test_transh(self, dataset, params):
        graph = Graph_ER(dataset)
        model = dataset()
        logger = get_logger('RL.Test.TransH.' + str(model))

        transh = TransH(graph, dimension=params['dimension'],
                        learning_rate=params['learning_rate'],
                        margin=params['margin'],
                        regularizer_scale=params['regularizer_scale'],
                        batchSize=params['batchSize'],
                        neg_rate=params['neg_rate'],
                        neg_rel_rate=params['neg_rel_rate'])
        loss = transh.train(max_epochs=params['epochs'])
        logger.info("Training Complete with loss: %f", loss)

        ent_embeddings = transh.get_ent_embeddings()
        result_prob = []
        for i in range(0, len(graph.entity_pairs)):
            distance = abs(spatial.distance.cosine(
                                ent_embeddings[graph.entity_pairs[i][0]],
                                ent_embeddings[graph.entity_pairs[i][1]]))
            result_prob.append((graph.entity_pairs[i][0], graph.entity_pairs[i][1], distance))
            #logger.info("i: %d, distance: %f true_pairs: %s", i, distance, graph.entity_pairs[i] in true_pairs)

        #Write Embeddings to file
        export_embeddings('er', str(model), 'TransH', graph.entity, ent_embeddings)
        export_result_prob(dataset, 'er', str(model), 'TransH', graph.entity, result_prob, graph.true_pairs)
        optimal_threshold, max_fscore = get_optimal_threshold(result_prob, graph.true_pairs)

        try:
            logger.info("MAX FSCORE: %f AT : %f", max_fscore, optimal_threshold)
            result = pd.MultiIndex.from_tuples([(e1, e2) for (e1, e2, d) in result_prob if d <= optimal_threshold])
            params['threshold'] = optimal_threshold
            log_quality_results(logger, result, graph.true_pairs, len(graph.entity_pairs), params)
        except:
            logger.info("Zero Reults")

        #Log MAP, MRR and Hits@K
        ir_metrics = InformationRetrievalMetrics(result_prob, graph.true_pairs)
        p_at_1 = ir_metrics.log_metrics(logger, params)

        transh.close_tf_session()
        return (max_fscore, p_at_1)

    def get_default_params(self):
        return {'learning_rate': 0.05, 'margin': 10, 'dimension': 128, 'epochs': 1000,
                'regularizer_scale' : 0.2, 'batchSize' : 1024, 'neg_rate' : 7, 'neg_rel_rate' : 1}

    def test_transh_cora(self):
        self._test_transh(Cora, self.get_default_params())

    def test_transh_febrl(self):
        self._test_transh(FEBRL, self.get_default_params())

    def test_transh_census(self):
        self._test_transh(Census, self.get_default_params())

    def _test_grid_search(self, dataset):
        dimension= [128]
        batchSize= [1024]
        learning_rate= [0.05, 0.1]
        margin= [1, 10]
        regularizer_scale = [0.05, 0.1]
        epochs = [1000]
        neg_rel_rate = [1, 7]
        neg_rate = [1, 4]

        model = dataset()
        logger = get_logger('RL.Test.GridSearch.TransH.' + str(model))
        count = 0
        max_fscore = 0
        max_prec_at_1 = 0

        for d, bs, lr, m, reg, e, nr, nrr in \
                itertools.product(dimension, batchSize, learning_rate, margin, regularizer_scale, epochs, neg_rate, neg_rel_rate):
            params = {'learning_rate': lr, 'margin': m, 'dimension': d, 'epochs': e, 'batchSize' : bs,
                        'regularizer_scale' : reg, 'neg_rate' : nr, 'neg_rel_rate' : nrr}
            logger.info("\nTest:%d PARAMS: %s", count, str(params))
            count = count + 1
            cur_fscore, cur_prec_at_1 = self._test_transh(dataset, params)
            if max_fscore <= cur_fscore:
                max_fscore = cur_fscore
            if max_prec_at_1 <= cur_prec_at_1:
                max_prec_at_1 = cur_prec_at_1

            logger.info("Ran total %d Tests.", count)
            logger.info("Max Fscore: %f", max_fscore)
            logger.info("Max Precision@1: %f", max_prec_at_1)

    def test_transh_grid_search_cora(self):
        self._test_grid_search(Cora)

    def test_transh_grid_search_febrl(self):
        self._test_grid_search(FEBRL)

    def test_transh_grid_search_census(self):
        self._test_grid_search(Census)