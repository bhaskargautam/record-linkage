import config
import itertools
import pandas as pd
import numpy as np
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
from ER.model import Graph_ER
from ER.transe import TransE
from scipy import spatial
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import precision_recall_curve

class TestRLTransE(unittest.TestCase):
    def _test_rl_transe(self, dataset, params):
        #Load Graph Data
        graph = Graph_ER(dataset)
        model = dataset()
        logger = get_logger('RL.Test.RLTransE.' + str(model))

        transe = TransE(graph, dimension=params['dimension'],
                        learning_rate=params['learning_rate'],
                        margin=params['margin'],
                        regularizer_scale=params['regularizer_scale'],
                        batchSize=params['batchSize'],
                        neg_rate=params['neg_rate'],
                        neg_rel_rate=params['neg_rel_rate'])
        loss = transe.train(max_epochs=params['epochs'])
        logger.info("Training Complete with loss: %f", loss)

        ent_embeddings = transe.get_ent_embeddings()

        result_prob = []
        for (a, b) in graph.entity_pairs:
            a_triples =  [(h, t, r) for (h, t, r) in graph.triples if h == a]
            b_triples =  [(h, t, r) for (h, t, r) in graph.triples if h == b]

            distance = abs(spatial.distance.cosine(ent_embeddings[a], ent_embeddings[b]))
            for (ah, at, ar) in a_triples:
                bt = [t for (h, t, r) in b_triples if r == ar]
                if len(bt):
                    distance = distance + abs(spatial.distance.cosine(\
                                            ent_embeddings[at], ent_embeddings[bt[0]]))
            result_prob.append((a, b, distance))
            #logger.info("a: %d, b: %d distance: %f true_pairs: %s", a, b, distance, (a, b) in graph.true_pairs)

        #Write Embeddings to file
        export_embeddings('er', str(model), 'RLTransE', graph.entity, ent_embeddings)
        export_result_prob(dataset, 'er', str(model), 'RLTransE', graph.entity, result_prob, graph.true_pairs)
        optimal_threshold, max_fscore = get_optimal_threshold(result_prob, graph.true_pairs, max_threshold=3.0)

        try:
            params['threshold'] = optimal_threshold
            result = pd.MultiIndex.from_tuples([(e1, e2) for (e1, e2, d) in result_prob if d <= optimal_threshold])
            log_quality_results(logger, result, graph.true_pairs, len(graph.entity_pairs), params)
        except:
            logger.info("Zero Reults")

        #Log MAP, MRR and Hits@K
        ir_metrics = InformationRetrievalMetrics(result_prob, graph.true_pairs)
        precison_at_1 = ir_metrics.log_metrics(logger, params)

        transe.close_tf_session()
        return (max_fscore, precison_at_1)

    def get_default_params(self):
        return {'learning_rate': 0.1, 'margin': 2, 'dimension': 128, 'epochs': 1000,
                'regularizer_scale' : 0.1, 'batchSize' : 1024, 'neg_rate' : 8, 'neg_rel_rate': 2}

    def test_cora(self):
        self._test_rl_transe(Cora, self.get_default_params())

    def test_febrl(self):
        self._test_rl_transe(FEBRL, self.get_default_params())

    def test_census(self):
        self._test_rl_transe(Census, self.get_default_params())