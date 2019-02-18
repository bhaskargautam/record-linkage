import config
import pandas as pd
import recordlinkage
import unittest

from common import (export_embeddings, get_optimal_threshold, get_logger,
            log_quality_results, sigmoid)
from data.cora import Cora
from data.febrl import FEBRL
from data.census import Census
from EAR.kr_ear import KR_EAR
from EAR.model import Graph_EAR
from scipy import spatial


logger = get_logger('KR_EAR')

class Test_KR_EAR(unittest.TestCase):

    def _test_kr_ear(self, dataset, file_prefix, params):
        logger = get_logger('Test_KR_EAR.' + str(file_prefix))
        try:
            graph = Graph_EAR(file_prefix)
            entity, attribute, relation, value, atriples, rtriples, \
                entity_pairs, true_pairs = graph.load_kg_ear_model()
        except IOError:
            model = dataset()
            entity, attribute, relation, value, atriples, rtriples, \
                entity_pairs, true_pairs = model.get_ear_model()

        kr_ear = KR_EAR(entity, attribute, relation, value, atriples, rtriples, entity_pairs,
                        dimension=params['dimension'],
                        learning_rate=params['learning_rate'],
                        margin=params['margin'],
                        regularizer_scale=params['regularizer_scale'],
                        batchSize=params['batchSize'],
                        neg_rate=params['neg_rate'],
                        neg_rel_rate=params['neg_rel_rate'])
        loss = kr_ear.train(max_epochs=params['epochs'])
        logger.info("Training Complete with loss: %f", loss)

        ent_embeddings = kr_ear.get_ent_embeddings()
        result_prob = []
        for i in range(0, len(entity_pairs)):
            distance = abs(spatial.distance.cosine(
                                ent_embeddings[entity_pairs[i][0]],
                                ent_embeddings[entity_pairs[i][1]]))
            result_prob.append((entity_pairs[i][0], entity_pairs[i][1], distance))
            #logger.info("i: %d, distance: %f true_pairs: %s", i, distance, entity_pairs[i] in true_pairs)

        #Write Embeddings to file
        export_embeddings(file_prefix, 'KR_EAR', entity, ent_embeddings)
        optimal_threshold, max_fscore = get_optimal_threshold(result_prob, true_pairs)

        try:
            logger.info("MAX FSCORE: %f AT : %f", max_fscore, optimal_threshold)
            result = pd.MultiIndex.from_tuples([(e1, e2) for (e1, e2, d) in result_prob if d <= optimal_threshold])
            params['threshold'] = optimal_threshold
            log_quality_results(logger, result, true_pairs, len(entity_pairs), params)
        except:
            logger.info("Zero Reults")
        kr_ear.close_tf_session()

    def get_default_params(self):
        return {'learning_rate': 0.1, 'margin': 1, 'dimension': 80, 'epochs': 500,
                'regularizer_scale' : 0.1, 'batchSize' : 100, 'neg_rate' : 10, 'neg_rel_rate' : 1}

    def test_krear_cora(self):
        self._test_kr_ear(Cora, config.CORA_FILE_PREFIX, self.get_default_params())

    def test_krear_febrl(self):
        self._test_kr_ear(FEBRL, config.FEBRL_FILE_PREFIX, self.get_default_params())

    def test_krear_census(self):
        self._test_kr_ear(Census, config.CENSUS_FILE_PREFIX, self.get_default_params())

    def _test_grid_search(self, model):
        dimension= [50, 80, 120]
        batchSize= [100]
        learning_rate= [0.1, 0.2]
        margin= [0, 0.5, 1]
        regularizer_scale = [0.1, 0.2]
        epochs = [100, 500]
        neg_rel_rate = [1, 2, 5]
        neg_rate = [1, 5, 10]

        logger = get_logger('Test_KR_EAR_GridSearch.' + str(model))
        count = 0
        max_fscore = 0
        for d, bs, lr, m, reg, e, nr, nrr in \
                itertools.product(dimension, batchSize, learning_rate, margin, regularizer_scale, epochs, neg_rate, neg_rel_rate):
            params = {'learning_rate': lr, 'margin': m, 'dimension': d, 'epochs': e, 'batchSize' : bs,
                            'regularizer_scale' : reg, 'neg_rate' : nr, 'neg_rel_rate' : nrr}
            logger.info("\nPARAMS: %s", str(params))
            count = count + 1
            cur_fscore = self._test_kr_ear(model, params)
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