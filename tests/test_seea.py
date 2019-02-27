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

    def _test_seea(self, dataset, file_prefix, params):
        logger = get_logger('RL.Test.SEEA.' + str(file_prefix))
        try:
            graph = Graph_EAR(file_prefix)
            entity, attribute, relation, value, atriples, rtriples, \
                entity_pairs, true_pairs = graph.load_kg_ear_model()
        except IOError:
            model = dataset()
            entity, attribute, relation, value, atriples, rtriples, \
                entity_pairs, true_pairs = model.get_ear_model()

        seea = SEEA(entity, attribute, relation, value, atriples, rtriples, entity_pairs,
                        dimension = params['dimension'],
                        learning_rate = params['learning_rate'],
                        batchSize = params['batchSize'],
                        margin = params['margin'],
                        regularizer_scale = params['regularizer_scale'],
                        neg_rate = params['neg_rate'],
                        neg_rel_rate = params['neg_rel_rate'])

        #Begin SEEA iterations, passing true pairs only to debug the alignments.
        results = seea.seea_iterate(entity_pairs, true_pairs, params['beta'],
                                    params['max_iter'], params['max_epochs'])
        try:
            result_pairs = pd.MultiIndex.from_tuples(results)
            log_quality_results(logger, result_pairs, true_pairs, len(entity_pairs), params)
        except Exception as e:
            logger.error(e)
            logger.info("No Aligned pairs found.")

        ent_embeddings = seea.get_ent_embeddings()
        export_embeddings('ear', file_prefix, 'SEEA', entity, ent_embeddings)

        result_prob = []
        for (e1, e2) in entity_pairs:
            distance = abs(spatial.distance.cosine(ent_embeddings[e1], ent_embeddings[e2]))
            result_prob.append((e1, e2, distance))
        export_result_prob(dataset, 'ear', file_prefix, 'SEEA', entity, result_prob, true_pairs)

        #Log MAP, MRR and Hits@K
        ir_metrics = InformationRetrievalMetrics(result_prob, true_pairs)
        ir_metrics.log_metrics(logger)

        seea.close_tf_session()

    def get_default_params(self):
        return {'beta': 10, 'max_iter' : 10, 'dimension': 80, 'learning_rate' : 0.1, 'batchSize' : 10,
                'margin' : 1, 'regularizer_scale' : 0.1, 'max_epochs' : 100, 'neg_rate' : 10, 'neg_rel_rate': 0}

    def test_seea_cora(self):
        self._test_seea(Cora, config.CORA_FILE_PREFIX, self.get_default_params())

    def test_seea_febrl(self):
        self._test_seea(FEBRL, config.FEBRL_FILE_PREFIX, self.get_default_params())

    def test_seea_census(self):
        self._test_seea(Census, config.CENSUS_FILE_PREFIX, self.get_default_params())

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
        self._test_grid_search(Cora, config.CORA_FILE_PREFIX)

    def test_grid_search_febrl(self):
        self._test_grid_search(FEBRL, config.FEBRL_FILE_PREFIX)

    def test_grid_search_census(self):
        self._test_grid_search(Census, config.CENSUS_FILE_PREFIX)