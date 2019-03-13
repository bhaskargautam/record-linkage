import config
import itertools
import pandas as pd
import unittest

from common import (
    export_embeddings,
    export_result_prob,
    get_logger,
    InformationRetrievalMetrics,
    log_quality_results)
from data.cora import Cora
from data.febrl import FEBRL
from data.census import Census
from EAR.seea import SEEA
from EAR.model import Graph_EAR
from scipy import spatial

class Test_SEEA(unittest.TestCase):

    def _test_seea(self, dataset, params):
        model = dataset()
        graph = Graph_EAR(dataset)
        logger = get_logger('RL.Test.SEEA.' + str(model))

        seea = SEEA(graph, dimension = params['dimension'],
                        learning_rate = params['learning_rate'],
                        batchSize = params['batchSize'],
                        margin = params['margin'],
                        regularizer_scale = params['regularizer_scale'],
                        neg_rate = params['neg_rate'],
                        neg_rel_rate = params['neg_rel_rate'])

        #Begin SEEA iterations, passing true pairs only to debug the alignments.
        results = seea.seea_iterate(beta = params['beta'],
                                    max_iter = params['max_iter'],
                                    max_epochs = params['max_epochs'])
        try:
            result_pairs = pd.MultiIndex.from_tuples(results)
            log_quality_results(logger, result_pairs, graph.true_pairs, len(graph.entity_pairs), params)
        except Exception as e:
            logger.error(e)
            logger.info("No Aligned pairs found.")

        ent_embeddings = seea.get_ent_embeddings()
        export_embeddings('ear', str(model), 'SEEA', graph.entity, ent_embeddings)

        result_prob = []
        for (e1, e2) in graph.entity_pairs:
            distance = abs(spatial.distance.cosine(ent_embeddings[e1], ent_embeddings[e2]))
            result_prob.append((e1, e2, distance))
        export_result_prob(dataset, 'ear', str(model), 'SEEA', graph.entity, result_prob, graph.true_pairs)

        #Log MAP, MRR and Hits@K
        ir_metrics = InformationRetrievalMetrics(result_prob, graph.true_pairs)
        ir_metrics.log_metrics(logger, params)

        seea.close_tf_session()

    def get_default_params(self):
        return {'beta': 100, 'max_iter' : 100, 'dimension': 80, 'learning_rate' : 0.1, 'batchSize' : 100,
                'margin' : 1, 'regularizer_scale' : 0.1, 'max_epochs' : 100, 'neg_rate' : 10, 'neg_rel_rate': 0}

    def test_seea_cora(self):
        self._test_seea(Cora, self.get_default_params())

    def test_seea_febrl(self):
        self._test_seea(FEBRL, self.get_default_params())

    def test_seea_census(self):
        self._test_seea(Census, self.get_default_params())

    def _test_grid_search(self, model, file_prefix):
        beta = [6, 10]
        dimension= [80, 120]
        batchSize= [100]
        learning_rate= [0.1, 0.2]
        margin= [10, 1]
        regularizer_scale = [0.1, 0.2]
        epochs = [100, 500]
        neg_rel_rate = [1, 2]
        neg_rate = [1, 10]
        max_iter = [10, 100]
        count = 0
        max_fscore = 0

        logger = get_logger('RL.Test.GridSearch.SEEA.' + str(file_prefix))

        for b, d, bs, lr, m, reg, e, nr, nrr, mi in \
            itertools.product(beta, dimension, batchSize, learning_rate, margin, regularizer_scale, epochs, neg_rate, neg_rel_rate, max_iter):
            params = {'beta': b, 'learning_rate': lr, 'margin': m, 'dimension': d, 'epochs': e, 'batchSize' : bs,
                        'regularizer_scale' : reg, 'neg_rate' : nr, 'neg_rel_rate': nrr, 'max_iter' : mi}
            logger.info("\nPARAMS: %s", str(params))
            count = count + 1
            cur_fscore = self._test_seea(model, file_prefix, params)
            if max_fscore <= cur_fscore:
                max_fscore = cur_fscore

        logger.info("Ran total %d Tests.", count)
        logger.info("Max Fscore: %f", max_fscore)

    def test_grid_search_cora(self):
        self._test_grid_search(Cora)

    def test_grid_search_febrl(self):
        self._test_grid_search(FEBRL)

    def test_grid_search_census(self):
        self._test_grid_search(Census)