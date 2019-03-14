import unittest
import copy

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
from ERER.mtranse import MTransE
from scipy import spatial

class TestMTransE(unittest.TestCase):

    def _test_mtranse(self, dataset, params):
        model = dataset()
        graph = Graph_ERER(dataset)
        logger = get_logger('RL.Test.MTransE.' + str(model))

        mtranse = MTransE(graph, dimension=params['dimension'],
                                 batchSize=params['batchSize'],
                                 learning_rate=params['learning_rate'],
                                 regularizer_scale = params['regularizer_scale'])
        mtranse.train(max_epochs=params['max_epochs'])

        ent_embeddings_a = mtranse.get_ent_embeddings_A()
        ent_embeddings_b = mtranse.get_ent_embeddings_B()
        result_prob = []
        for i in range(0, len(graph.entity_pairs)):
            distance = abs(spatial.distance.cosine(
                                ent_embeddings_a[int(graph.entity_pairs[i][0])],
                                ent_embeddings_b[int(graph.entity_pairs[i][1])]))
            result_prob.append((graph.entity_pairs[i][0], graph.entity_pairs[i][1], distance))
            #logger.info("i: %d, distance: %f true_pairs: %s", i, distance, graph.entity_pairs[i] in true_pairs)

        #Write Embeddings to file
        export_embeddings('erer', str(model), 'MTransE', graph.entityA, ent_embeddings_a)
        export_embeddings('erer', str(model), 'MTransE', graph.entityB, ent_embeddings_b)
        export_result_prob(dataset, 'erer', str(model), 'MTransE', graph.entityA, result_prob, graph.true_pairs, graph.entityB)
        optimal_threshold, max_fscore = get_optimal_threshold(result_prob, graph.true_pairs)

        try:
            params['threshold'] = optimal_threshold
            result = pd.MultiIndex.from_tuples([(e1, e2) for (e1, e2, d) in result_prob if d <= optimal_threshold])
            log_quality_results(logger, result, graph.true_pairs, len(graph.entity_pairs), params)
        except:
            logger.info("Zero Reults")

        #Log MAP, MRR and Hits@K
        ir_metrics = InformationRetrievalMetrics(result_prob, graph.true_pairs)
        ir_metrics.log_metrics(logger, params)

        mtranse.close_tf_session()
        return max_fscore

    def get_default_params(self):
        return {'learning_rate': 0.1, 'dimension': 80, 'max_epochs': 500,
                'regularizer_scale' : 0.1, 'batchSize' : 100}

    def test_cora(self):
        self._test_mtranse(Cora, self.get_default_params())

    def test_febrl(self):
        self._test_mtranse(FEBRL, self.get_default_params())

    def test_census(self):
        self._test_mtranse(Census, self.get_default_params())