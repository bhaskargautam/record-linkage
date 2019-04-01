import copy
import itertools
import pandas as pd
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
from ERER.model import Graph_ERER
from ERER.etranse import ETransE
from scipy import spatial


class TestETransE(unittest.TestCase):

    def _test_etranse(self, dataset, params):
        model = dataset()
        graph = Graph_ERER(dataset)
        logger = get_logger("RL.Test.ETransE." + str(model))

        etranse = ETransE(graph, dimension=params['dimension'],
                                batchSize=params['batchSize'],
                                learning_rate=params['learning_rate'],
                                margin=params['margin'],
                                neg_rate=params['neg_rate'],
                                neg_rel_rate=params['neg_rel_rate'],
                                regularizer_scale=params['regularizer_scale'],
                                alpha=params['alpha'],
                                beta=params['beta'])
        etranse.train(max_epochs = params['max_epochs'])
        ent_embeddings_a = etranse.get_ent_embeddings_A()
        ent_embeddings_b = etranse.get_ent_embeddings_B()

        result_prob = []
        for i in range(0, len(graph.entity_pairs)):
            distance = abs(spatial.distance.cosine(
                                ent_embeddings_a[int(graph.entity_pairs[i][0])],
                                ent_embeddings_b[int(graph.entity_pairs[i][1])]))
            result_prob.append((graph.entity_pairs[i][0], graph.entity_pairs[i][1], distance))
            #logger.info("i: %d, distance: %f true_pairs: %s", i, distance, graph.entity_pairs[i] in true_pairs)

        #Write Embeddings to file
        export_embeddings('erer', str(model), 'ETransE', graph.entityA, ent_embeddings_a)
        export_embeddings('erer', str(model), 'ETransE', graph.entityB, ent_embeddings_b)
        export_result_prob(dataset, 'erer', str(model), 'ETransE', graph.entityA, result_prob, graph.true_pairs, graph.entityB)
        optimal_threshold, max_fscore = get_optimal_threshold(result_prob, graph.true_pairs)

        try:
            params['threshold'] = optimal_threshold
            result = pd.MultiIndex.from_tuples([(e1, e2) for (e1, e2, d) in result_prob if d <= optimal_threshold])
            log_quality_results(logger, result, graph.true_pairs, len(graph.entity_pairs), params)
        except Exception as e:
            logger.info("Zero Reults")
            logger.error(e)

        #Log MAP, MRR and Hits@K
        ir_metrics = InformationRetrievalMetrics(result_prob, graph.true_pairs)
        prec_at_1 = ir_metrics.log_metrics(logger, params)

        etranse.close_tf_session()
        return (max_fscore, prec_at_1)


    def get_default_params(self):
        return { "dimension": 64, "batchSize": 128, "learning_rate": 0.1, 'max_epochs':100,
                "margin": 1, "neg_rate": 7, "neg_rel_rate": 1, "regularizer_scale": 0.1,
                'alpha' : 5, 'beta' : 5}

    def test_cora(self):
        self._test_etranse(Cora, self.get_default_params())

    def test_febrl(self):
        self._test_etranse(FEBRL, self.get_default_params())

    def test_census(self):
        self._test_etranse(Census, self.get_default_params())