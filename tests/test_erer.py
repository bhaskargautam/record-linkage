import config
import itertools
import pandas as pd
import recordlinkage
import unittest
from common import (
    export_embeddings,
    export_result_prob,
    get_logger,
    get_optimal_threshold,
    log_quality_results)
from data.cora import Cora
from data.febrl import FEBRL
from data.census import Census
from ERER.model import Graph_ERER
from ER.transe import TransE
from ER.transh import TransH
from scipy import spatial
from sklearn.metrics import precision_recall_curve

class TestERER(unittest.TestCase):
    def _test_erer(self, dataset, file_prefix, er_algo, params):
        try:
            graph = Graph_ERER(file_prefix)
            graph.load_kg_erer_model()
            entity, relation, triples, entity_pairs, true_pairs = graph.get_er_model()
        except IOError:
            model = dataset()
            entA, entB, relA, relB, triA, triB, entity_pairs, prior_pairs, true_pairs = model.get_erer_model()
            graph = Graph_ERER(str(model))
            graph.export_kg_erer_model(entA, entB, relA, relB, triA, triB, entity_pairs, prior_pairs, true_pairs)
            entity, relation, triples, entity_pairs, true_pairs = graph.get_er_model()

        er_model = er_algo(entity, relation, triples, entity_pairs,
                        dimension=params['dimension'],
                        learning_rate=params['learning_rate'],
                        margin=params['margin'],
                        regularizer_scale=params['regularizer_scale'],
                        batchSize=params['batchSize'],
                        neg_rate=params['neg_rate'],
                        neg_rel_rate=params['neg_rel_rate'])
        loss = er_model.train(max_epochs=params['epochs'])

        logger = get_logger('RL.Test.ERER.' + file_prefix + "." + str(er_model))
        logger.info("Training Complete with loss: %f", loss)

        ent_embeddings = er_model.get_ent_embeddings()
        result_prob = []
        for i in range(0, len(entity_pairs)):
            distance = abs(spatial.distance.cosine(
                                ent_embeddings[entity_pairs[i][0]],
                                ent_embeddings[entity_pairs[i][1]]))
            result_prob.append((entity_pairs[i][0], entity_pairs[i][1], distance))
            #logger.info("i: %d, distance: %f true_pairs: %s", i, distance, entity_pairs[i] in true_pairs)

        #Write Embeddings to file
        export_embeddings("erer", file_prefix, str(er_model), entity, ent_embeddings)
        export_result_prob(dataset, 'erer', file_prefix, str(er_model), entity, result_prob, true_pairs)

        optimal_threshold, max_fscore = get_optimal_threshold(result_prob, true_pairs)

        try:
            params['threshold'] = optimal_threshold
            result = pd.MultiIndex.from_tuples([(e1, e2) for (e1, e2, d) in result_prob if d <= optimal_threshold])
            log_quality_results(logger, result, true_pairs, len(entity_pairs), params)
        except:
            logger.info("Zero Reults")

        er_model.close_tf_session()
        return max_fscore


    def get_default_params(self):
        return {'learning_rate': 0.1, 'margin': 1, 'dimension': 80, 'epochs': 500,
                'regularizer_scale' : 0.1, 'batchSize' : 100, 'neg_rate' : 8, 'neg_rel_rate': 2}

    def test_erer_transe_cora(self):
        self._test_erer(Cora, config.CORA_FILE_PREFIX, TransE, self.get_default_params())

    def test_erer_transh_cora(self):
        self._test_erer(Cora, config.CORA_FILE_PREFIX, TransH, self.get_default_params())

    def test_erer_transe_febrl(self):
        self._test_erer(FEBRL, config.FEBRL_FILE_PREFIX, TransE, self.get_default_params())

    def test_erer_transh_febrl(self):
        self._test_erer(FEBRL, config.FEBRL_FILE_PREFIX, TransH, self.get_default_params())

    def test_erer_transe_census(self):
        self._test_erer(Census, config.CENSUS_FILE_PREFIX, TransE, self.get_default_params())

    def test_erer_transh_census(self):
        self._test_erer(Census, config.CENSUS_FILE_PREFIX, TransH, self.get_default_params())