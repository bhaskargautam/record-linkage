import config
import pandas as pd
import unittest

from common import get_logger, log_quality_results, sigmoid
from data.cora import Cora
from data.febrl import FEBRL
from data.census import Census
from ER.transe import TransE
from scipy import spatial
from sklearn.metrics import precision_recall_curve
import recordlinkage

logger = get_logger('TestTransE')

class TestTransE(unittest.TestCase):
    def _test_transe(self, dataset, params):
        model = dataset()
        logger = get_logger('TestTransE.' + str(model))

        entity, relation, triples, entity_pairs, true_pairs = model.get_er_model()
        transe = TransE(entity, relation, triples,
                        dimension=params['dimension'],
                        learning_rate=params['learning_rate'],
                        margin=params['margin'])
        loss = transe.train(max_epochs=params['epochs'])
        logger.info("Training Complete with loss: %f", loss)

        ent_embeddings = transe.get_ent_embeddings()
        logger.info(ent_embeddings[true_pairs[0][0]])
        logger.info(ent_embeddings[true_pairs[0][1]])
        result_prob = []
        for i in range(0, len(entity_pairs)):
            distance = abs(spatial.distance.cosine(
                                ent_embeddings[entity_pairs[i][0]],
                                ent_embeddings[entity_pairs[i][1]]))
            result_prob.append((entity_pairs[i][0], entity_pairs[i][1], distance))
            #logger.info("i: %d, distance: %f true_pairs: %s", i, distance, entity_pairs[i] in true_pairs)

        max_fscore = 0.0
        optimal_threshold = 0.0
        for threshold in range(0, 80, 5):
            threshold = threshold / 100.0
            logger.info("\nTesting for threshold: %f", threshold)
            try:
                result = pd.MultiIndex.from_tuples([(e1, e2) for (e1, e2, d) in result_prob if d <= threshold])
                params['threshold'] = threshold
                log_quality_results(logger, result, true_pairs, len(entity_pairs), params)

                fscore = recordlinkage.fscore(true_pairs, result)
                if fscore >= max_fscore:
                    max_fscore = fscore
                    optimal_threshold = threshold
            except:
                logger.info("No results")
        try:
            logger.info("MAX FSCORE: %f AT : %f", max_fscore, optimal_threshold)
            params['threshold'] = optimal_threshold
            result = pd.MultiIndex.from_tuples([(e1, e2) for (e1, e2, d) in result_prob if d <= optimal_threshold])
            log_quality_results(logger, result, true_pairs, len(entity_pairs), params)
        except:
            logger.info("Zero Reults")

        transe.close_tf_session()

    @unittest.skip("Takes Tooo long")
    def test_transe_cora_grid_search(self):
        params = {
            'learning_rate': 0.1,
            'dimension' : 80,
            'margin' : 1,
            'epochs' : 100
        }
        for learning_rate in range(1, 9):
            learning_rate = learning_rate / 10.0
            params['learning_rate'] = learning_rate
            for dimension in range(10, 100, 10):
                params['dimension'] = dimension
                for margin in range(1, 10):
                    margin = margin / 10.0
                    params['margin'] = margin
                    self._test_transe(Cora, params)

    def test_transe_cora(self):
        self._test_transe(Cora, {'learning_rate': 0.1, 'margin': 1, 'dimension': 80, 'epochs': 500})

    def test_transe_febrl(self):
        self._test_transe(FEBRL, {'learning_rate': 0.1, 'margin': 1, 'dimension': 80, 'epochs': 500})

    def test_transe_census(self):
        self._test_transe(Census, {'learning_rate': 0.1, 'margin': 1, 'dimension': 80, 'epochs': 500})