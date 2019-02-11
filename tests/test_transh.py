import config
import itertools
import pandas as pd
import recordlinkage
import unittest

from common import get_logger, log_quality_results
from data.cora import Cora
from data.febrl import FEBRL
from data.census import Census
from ER.transh import TransH
from scipy import spatial

class TestTransH(unittest.TestCase):

    def _test_transh(self, dataset, params):
        model = dataset()
        logger = get_logger('TestTransH.' + str(model))
        entity, relation, triples, entity_pairs, true_pairs = model.get_er_model()
        transh = TransH(entity, relation, triples,dimension=params['dimension'],
                        learning_rate=params['learning_rate'],
                        margin=params['margin'],
                        regularizer_scale=params['regularizer_scale'],
                        batchSize=params['batchSize'])
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

        #Write cosine distance to file for analysis
        """
        with open('cora_transh.txt', 'w+') as f:
            for (e1, e2, d) in result_prob:
                f.write("%d, %d, %f, %s\n" % (e1, e2, d, (e1, e2) in true_pairs))
        """

        max_fscore = 0.0
        optimal_threshold = 0
        for threshold in range(20,110, 5):
            threshold = threshold / 100.0
            logger.info("\n\n Testing for threshold: %f", threshold)
            try:
                result = pd.MultiIndex.from_tuples([(e1, e2) for (e1, e2, d) in result_prob if d <= threshold])
                params['threshold'] = threshold
                log_quality_results(logger, result, true_pairs, len(entity_pairs), params)

                fscore = recordlinkage.fscore(true_pairs, result)
                if fscore >= max_fscore:
                    max_fscore = fscore
                    optimal_threshold = threshold
            except Exception as e:
                logger.info("No results")
                logger.info(e)
        try:
            logger.info("MAX FSCORE: %f AT : %f", max_fscore, optimal_threshold)
            result = pd.MultiIndex.from_tuples([(e1, e2) for (e1, e2, d) in result_prob if d <= optimal_threshold])
            params['threshold'] = optimal_threshold
            log_quality_results(logger, result, true_pairs, len(entity_pairs), params)
        except:
            logger.info("Zero Reults")

        transh.close_tf_session()
        return max_fscore

    def get_default_params(self):
        return {'learning_rate': 0.1, 'margin': 1, 'dimension': 80, 'epochs': 500,
                'regularizer_scale' : 0.1, 'batchSize' : 100}

    def test_transh_cora(self):
        self._test_transh(Cora, self.get_default_params())

    def test_transh_febrl(self):
        self._test_transh(FEBRL, self.get_default_params())

    def test_transh_census(self):
        self._test_transh(Census, self.get_default_params())

    def _test_grid_search(self, model):
        dimension= [50, 80, 120]
        batchSize= [100]
        learning_rate= [0.1, 0.2]
        margin= [0, 0.5, 1]
        regularizer_scale = [0.1, 0.2]
        epochs = [100, 500]

        logger = get_logger('TestGridSearchTransH.' + str(model))
        count = 0
        max_fscore = 0
        for d, bs, lr, m, reg, e in itertools.product(dimension, batchSize, learning_rate, margin, regularizer_scale, epochs):
            params = {'learning_rate': lr, 'margin': m, 'dimension': d, 'epochs': e, 'batchSize' : bs, 'regularizer_scale' : reg}
            logger.info("\nPARAMS: %s", str(params))
            count = count + 1
            cur_fscore = self._test_transh(model, params)
            if max_fscore <= cur_fscore:
                max_fscore = cur_fscore

        logger.info("Ran total %d Tests.", count)
        logger.info("Max Fscore: %f", max_fscore)

    def test_transh_grid_search_cora(self):
        self._test_grid_search(Cora)

    def test_transh_grid_search_febrl(self):
        self._test_grid_search(FEBRL)

    def test_transh_grid_search_census(self):
        self._test_grid_search(Census)