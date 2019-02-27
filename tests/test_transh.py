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

    def _test_transh(self, dataset, file_prefix, params):
        logger = get_logger('RL.Test.TransH.' + str(file_prefix))
        try:
            graph = Graph_ER(file_prefix)
            entity, relation, triples, entity_pairs, true_pairs = graph.load_kg_er_model()
        except IOError:
            model = dataset()
            entity, relation, triples, entity_pairs, true_pairs = model.get_er_model()
        transh = TransH(entity, relation, triples, entity_pairs,
                        dimension=params['dimension'],
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
        for i in range(0, len(entity_pairs)):
            distance = abs(spatial.distance.cosine(
                                ent_embeddings[entity_pairs[i][0]],
                                ent_embeddings[entity_pairs[i][1]]))
            result_prob.append((entity_pairs[i][0], entity_pairs[i][1], distance))
            #logger.info("i: %d, distance: %f true_pairs: %s", i, distance, entity_pairs[i] in true_pairs)

        #Write Embeddings to file
        export_embeddings('er', file_prefix, 'TransH', entity, ent_embeddings)
        export_result_prob(dataset, 'er', file_prefix, 'TransH', entity, result_prob, true_pairs)
        optimal_threshold, max_fscore = get_optimal_threshold(result_prob, true_pairs)

        try:
            logger.info("MAX FSCORE: %f AT : %f", max_fscore, optimal_threshold)
            result = pd.MultiIndex.from_tuples([(e1, e2) for (e1, e2, d) in result_prob if d <= optimal_threshold])
            params['threshold'] = optimal_threshold
            log_quality_results(logger, result, true_pairs, len(entity_pairs), params)
        except:
            logger.info("Zero Reults")

        #Log MAP, MRR and Hits@K
        ir_metrics = InformationRetrievalMetrics(result_prob, true_pairs)
        ir_metrics.log_metrics(logger)

        transh.close_tf_session()
        return max_fscore

    def get_default_params(self):
        return {'learning_rate': 0.1, 'margin': 1, 'dimension': 80, 'epochs': 500,
                'regularizer_scale' : 0.1, 'batchSize' : 100, 'neg_rate' : 10, 'neg_rel_rate' : 2}

    def test_transh_cora(self):
        self._test_transh(Cora, config.CORA_FILE_PREFIX, self.get_default_params())

    def test_transh_febrl(self):
        self._test_transh(FEBRL, config.FEBRL_FILE_PREFIX, self.get_default_params())

    def test_transh_census(self):
        self._test_transh(Census, config.CENSUS_FILE_PREFIX, self.get_default_params())

    def _test_grid_search(self, model, file_prefix):
        dimension= [50, 80, 120]
        batchSize= [100]
        learning_rate= [0.1, 0.2]
        margin= [0.5, 1]
        regularizer_scale = [0.1, 0.2]
        epochs = [100, 500]
        neg_rel_rate = [1, 2, 5]
        neg_rate = [1, 5, 10]

        logger = get_logger('RL.Test.GridSearch.TransH.' + str(model))
        count = 0
        max_fscore = 0
        for d, bs, lr, m, reg, e, nr, nrr in \
                itertools.product(dimension, batchSize, learning_rate, margin, regularizer_scale, epochs, neg_rate, neg_rel_rate):
            params = {'learning_rate': lr, 'margin': m, 'dimension': d, 'epochs': e, 'batchSize' : bs,
                        'regularizer_scale' : reg, 'neg_rate' : nr, 'neg_rel_rate' : nrr}
            logger.info("\nPARAMS: %s", str(params))
            count = count + 1
            cur_fscore = self._test_transh(model, file_prefix, params)
            if max_fscore <= cur_fscore:
                max_fscore = cur_fscore

        logger.info("Ran total %d Tests.", count)
        logger.info("Max Fscore: %f", max_fscore)

    def test_transh_grid_search_cora(self):
        self._test_grid_search(Cora, config.CORA_FILE_PREFIX)

    def test_transh_grid_search_febrl(self):
        self._test_grid_search(FEBRL, config.FEBRL_FILE_PREFIX)

    def test_transh_grid_search_census(self):
        self._test_grid_search(Census, config.CENSUS_FILE_PREFIX)