import config
import itertools
import pandas as pd
import numpy as np
import recordlinkage
import unittest

from common import (
    export_embeddings,
    export_false_positives,
    export_false_negatives,
    export_result_prob,
    get_optimal_threshold,
    get_logger,
    InformationRetrievalMetrics,
    log_quality_results,
    sigmoid)
from data.cora import Cora
from data.febrl import FEBRL
from data.census import Census
from veer import VEER
from scipy import spatial
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import precision_recall_curve

class TestVEER(unittest.TestCase):
    def _test_veer(self, model, columns, params):
        #Load Graph Data
        dataset = model()
        logger = get_logger('RL.Test.VEER.' + str(dataset))

        veer = VEER(model, columns, dimension=params['dimension'],
                        learning_rate=params['learning_rate'],
                        margin=params['margin'],
                        regularizer_scale=params['regularizer_scale'],
                        batchSize=params['batchSize'])

        #Train Model
        loss, val_loss = veer.train(max_epochs=params['epochs'])
        logger.info("Training Complete with loss: %f, val_loss:%f", loss, val_loss)

        #Test Model
        result_prob, accuracy = veer.test()
        logger.info("Predict count: %d", len(result_prob))
        logger.info("Sample Prob: %s", str([ (c, (a, b) in dataset.true_test_links)
                                        for (a,b,c) in result_prob[:20]]))
        logger.info("Column Weights: %s", str(veer.get_col_weights()))
        logger.info("Accuracy: %s", str(accuracy))
        logger.info("Sample embeddings: %s", str(veer.get_val_embeddings()[0]))

        #Compute Performance measures
        optimal_threshold, max_fscore = get_optimal_threshold(result_prob, dataset.true_test_links, max_threshold=2.0)

        try:
            params['threshold'] = optimal_threshold
            result = pd.MultiIndex.from_tuples([(e1, e2) for (e1, e2, d) in result_prob if d <= optimal_threshold])
            log_quality_results(logger, result, dataset.true_test_links, len(dataset.test_links), params)
        except Exception as e:
            logger.info("Zero Reults")
            logger.error(e)

        #Log MAP, MRR and Hits@K
        ir_metrics = InformationRetrievalMetrics(result_prob, dataset.true_test_links)
        precison_at_1 = ir_metrics.log_metrics(logger, params)

        #Write Result Prob to file
        entitiesA = dataset.get_entity_names(dataset.testDataA)
        entitiesB = dataset.get_entity_names(dataset.testDataB)
        index_dictA = {str(dataset.testDataA.iloc[i]._name) : i
                        for i in range(dataset.testDataA.shape[0])}
        index_dictB = {str(dataset.testDataB.iloc[i]._name) : i
                        for i in range(dataset.testDataB.shape[0])}
        result_prob = [(index_dictA[str(a)], index_dictB[str(b)], p)
                            for (a, b, p) in result_prob]
        export_result_prob(dataset, 'veg', str(dataset), 'VEER', entitiesA, result_prob,
                                    dataset.true_test_links, entitiesB)
        export_false_negatives(model, 'veg', str(dataset), 'VEER', entitiesA, result_prob,
                            dataset.true_test_links, result, entitiesB)
        export_false_positives(model, 'veg', str(dataset), 'VEER', entitiesA, result_prob,
                            dataset.true_test_links, result, entitiesB)

        veer.close_tf_session()
        return (max_fscore, precison_at_1)

    def get_default_params(self):
        return {'learning_rate': 0.1, 'margin': 0.1, 'dimension': 16, 'epochs': 30,
                'regularizer_scale' : 0.1, 'batchSize' : 512 }

    def test_veer_cora(self):
        self._test_veer(Cora, ['title', 'author', 'publisher', 'date',
                        'pages', 'volume', 'journal', 'address'],
                        self.get_default_params())

    def test_veer_febrl(self):
        self._test_veer(FEBRL, ['surname', 'state', 'date_of_birth', 'postcode'],
                self.get_default_params())

    def test_veer_census(self):
        self._test_veer(Census, ['Noms_harmo', 'cognom_1', 'cohort', 'estat_civil',
                    'parentesc_har', 'ocupacio_hisco'], self.get_default_params())

    def _test_grid_search(self, dataset, columns):
        dimension= [128, 256]
        batchSize= [1024, 32]
        learning_rate= [0.1]
        margin= [1]
        regularizer_scale = [0.1]
        epochs = [1000, 5000]
        count = 0
        max_fscore = 0
        max_prec_at_1 = 0

        model = dataset()
        logger = get_logger('RL.Test.GridSearch.VEER.' + str(model))

        for d, bs, lr, m, reg, e in \
            itertools.product(dimension, batchSize, learning_rate, margin, regularizer_scale, epochs):
            params = {'learning_rate': lr, 'margin': m, 'dimension': d, 'epochs': e, 'batchSize' : bs,
                        'regularizer_scale' : reg, 'neg_rate' : nr, 'neg_rel_rate': nrr}
            logger.info("\nTest:%d, PARAMS: %s", count, str(params))
            count = count + 1

            cur_fscore, cur_prec_at_1 = self._test_veer(dataset, columns, params)
            if max_fscore <= cur_fscore:
                max_fscore = cur_fscore
            if max_prec_at_1 <= cur_prec_at_1:
                max_prec_at_1 = cur_prec_at_1

            logger.info("Ran total %d Tests.", count)
            logger.info("Max Fscore: %f", max_fscore)
            logger.info("Max Mean Precision@1: %f", max_prec_at_1)

    def test_grid_search_cora(self):
        self._test_grid_search(Cora, ['title', 'author', 'publisher', 'date',
                        'pages', 'volume', 'journal', 'address'])

    def test_grid_search_febrl(self):
        self._test_grid_search(FEBRL, ['surname', 'state', 'date_of_birth', 'postcode'])

    def test_grid_search_census(self):
        self._test_grid_search(Census, ['Noms_harmo', 'cognom_1', 'cohort', 'estat_civil',
                    'parentesc_har', 'ocupacio_hisco'])