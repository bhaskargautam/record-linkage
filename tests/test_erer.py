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
    InformationRetrievalMetrics,
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
    def _test_erer(self, dataset, er_algo, params):
        model = dataset()
        graph = Graph_ERER(dataset)
        graph_er = graph.get_er_model()

        er_model = er_algo(graph_er,
                        dimension=params['dimension'],
                        learning_rate=params['learning_rate'],
                        margin=params['margin'],
                        regularizer_scale=params['regularizer_scale'],
                        batchSize=params['batchSize'],
                        neg_rate=params['neg_rate'],
                        neg_rel_rate=params['neg_rel_rate'])
        loss = er_model.train(max_epochs=params['epochs'])

        logger = get_logger('RL.Test.ERER.' + str(model) + "." + str(er_model))
        logger.info("Training Complete with loss: %f", loss)

        ent_embeddings = er_model.get_ent_embeddings()
        result_prob = []
        for i in range(0, len(graph_er.entity_pairs)):
            distance = abs(spatial.distance.cosine(
                                ent_embeddings[graph_er.entity_pairs[i][0]],
                                ent_embeddings[graph_er.entity_pairs[i][1]]))
            result_prob.append((graph_er.entity_pairs[i][0], graph_er.entity_pairs[i][1], distance))
            #logger.info("i: %d, distance: %f true_pairs: %s", i, distance, graph_er.entity_pairs[i] in graph_er.true_pairs)

        #Write Embeddings to file
        export_embeddings("erer", str(model), str(er_model), graph_er.entity, ent_embeddings)
        export_result_prob(dataset, 'erer', str(model), str(er_model), graph_er.entity, result_prob, graph_er.true_pairs)

        optimal_threshold, max_fscore = get_optimal_threshold(result_prob, graph_er.true_pairs)

        try:
            params['threshold'] = optimal_threshold
            result = pd.MultiIndex.from_tuples([(e1, e2) for (e1, e2, d) in result_prob if d <= optimal_threshold])
            log_quality_results(logger, result, graph_er.true_pairs, len(graph_er.entity_pairs), params)
        except:
            logger.info("Zero Reults")

        #Log MAP, MRR and Hits@K
        ir_metrics = InformationRetrievalMetrics(result_prob, graph_er.true_pairs)
        ir_metrics.log_metrics(logger)

        er_model.close_tf_session()
        return max_fscore


    def get_default_params(self):
        return {'learning_rate': 0.1, 'margin': 1, 'dimension': 80, 'epochs': 500,
                'regularizer_scale' : 0.1, 'batchSize' : 100, 'neg_rate' : 8, 'neg_rel_rate': 2}

    def test_erer_transe_cora(self):
        self._test_erer(Cora, TransE, self.get_default_params())

    def test_erer_transh_cora(self):
        self._test_erer(Cora, TransH, self.get_default_params())

    def test_erer_transe_febrl(self):
        self._test_erer(FEBRL, TransE, self.get_default_params())

    def test_erer_transh_febrl(self):
        self._test_erer(FEBRL, TransH, self.get_default_params())

    def test_erer_transe_census(self):
        self._test_erer(Census, TransE, self.get_default_params())

    def test_erer_transh_census(self):
        self._test_erer(Census, TransH, self.get_default_params())