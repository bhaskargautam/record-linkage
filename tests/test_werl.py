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
from ER.model import Graph_ER
from ER.transe import TransE
from werl import WERL
from scipy import spatial

class TestWERL(unittest.TestCase):

    def _get_embedding(self, ent_embeddings, columns, graph,
                dataA, dataB, candidate_pairs, true_pairs):
        logger = get_logger('RL.Test.WERL.GET_EMBEDDING.')

        embeddingA = []
        embeddingB = []
        truth = []
        zero_embedding = [0] * len(ent_embeddings[0])
        missing_vals = 0
        for (a, b) in candidate_pairs:
            row_a = dataA.loc[a]
            row_b = dataB.loc[b]

            embedA = []
            embedB = []
            for c in columns:
                try:
                    embedA.append(ent_embeddings[graph.entity.index(row_a[c])])
                except ValueError:
                    embedA.append(zero_embedding)
                    missing_vals = missing_vals + 1
                try:
                    embedB.append(ent_embeddings[graph.entity.index(row_b[c])])
                except ValueError:
                    embedB.append(zero_embedding)
                    missing_vals = missing_vals + 1

            embeddingA.append(embedA)
            embeddingB.append(embedB)
            truth.append((a, b) in true_pairs)

        logger.info("Size of embeddingA: %d, %d", len(embeddingA), len(embeddingA[0]))
        logger.info("Size of embeddingB: %d, %d", len(embeddingB), len(embeddingB[0]))
        logger.info("No. of missing_vals: %d", missing_vals)
        return (embeddingA, embeddingB, truth)

    def _test_werl(self, model, columns, params):
        #Load Graph Data
        dataset = model()
        graph = Graph_ER(model)
        logger = get_logger('RL.Test.WERL.' + str(dataset))

        #Train TransE embedding vectors
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
        transe.close_tf_session()

        #Train WERL weights
        werl = WERL(model, columns, graph.entity, ent_embeddings,
                        dimension=params['dimension'],
                        learning_rate=params['learning_rate'],
                        margin=params['margin'],
                        regularizer_scale=params['regularizer_scale'],
                        batchSize=params['batchSize'])
        loss, val_loss = werl.train(max_epochs=params['epochs'])
        logger.info("Training Complete with loss: %f, val_loss:%f", loss, val_loss)

        #Test Model
        result_prob, accuracy = werl.test()
        logger.info("Predict count: %d", len(result_prob))
        logger.info("Sample Prob: %s", str([ (c, (a, b) in dataset.true_test_links)
                                        for (a,b,c) in result_prob[:20]]))
        logger.info("Column Weights: %s", str(werl.get_col_weights()))
        logger.info("Accuracy: %s", str(accuracy))

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

        #Test Without Weights
        logger = get_logger('RL.Test.WERL.NoWT.' + str(dataset))

        result_prob, accuracy = werl.test_without_weight()
        logger.info("Predict count: %d", len(result_prob))
        logger.info("Sample Prob: %s", str([ (c, (a, b) in dataset.true_test_links)
                                        for (a,b,c) in result_prob[:20]]))
        logger.info("Column Weights: %s", str(werl.get_col_weights()))
        logger.info("Accuracy: %s", str(accuracy))

        #Compute Performance measures
        optimal_threshold, nowt_max_fscore = get_optimal_threshold(result_prob, dataset.true_test_links, max_threshold=2.0)

        try:
            params['threshold'] = optimal_threshold
            result = pd.MultiIndex.from_tuples([(e1, e2) for (e1, e2, d) in result_prob if d <= optimal_threshold])
            log_quality_results(logger, result, dataset.true_test_links, len(dataset.test_links), params)
        except Exception as e:
            logger.info("Zero Reults")
            logger.error(e)

        #Log MAP, MRR and Hits@K
        ir_metrics = InformationRetrievalMetrics(result_prob, dataset.true_test_links)
        nowt_precison_at_1 = ir_metrics.log_metrics(logger, params)
        werl.close_tf_session()

        return (max_fscore, precison_at_1)

    def get_default_params(self):
        return {'learning_rate': 0.1, 'margin': 0.1, 'dimension': 32, 'epochs': 1000,
                'regularizer_scale' : 0.1, 'batchSize' : 512, 'neg_rate' : 7, 'neg_rel_rate': 1}

    def test_cora(self):
        return self._test_werl(Cora, ['title', 'author', 'publisher', 'date',
                        'pages', 'volume', 'journal', 'address'], self.get_default_params())

    def test_febrl(self):
        self._test_werl(FEBRL, ['surname', 'state', 'date_of_birth', 'postcode'],
                self.get_default_params())

    def test_census(self):
        self._test_werl(Census, ['Noms_harmo', 'cognom_1', 'cohort', 'estat_civil',
                    'parentesc_har', 'ocupacio_hisco'], self.get_default_params())

    def _test_grid_search(self, dataset, columns):
        dimension= [16, 32, 64]
        batchSize= [32, 64]
        learning_rate= [0.1]
        margin= [1, 0.1]
        regularizer_scale = [0.1]
        epochs = [50, 100, 500]
        neg_rate = [7]
        neg_rel_rate = [1]
        count = 0
        max_fscore = 0
        max_prec_at_1 = 0

        model = dataset()
        logger = get_logger('RL.Test.GridSearch.WERL.' + str(model))

        for d, bs, lr, m, reg, e, nr, nrr in itertools.product(dimension, batchSize,
                learning_rate, margin, regularizer_scale, epochs, neg_rate, neg_rel_rate):
            params = {'learning_rate': lr, 'margin': m, 'dimension': d, 'epochs': e, 'batchSize' : bs,
                        'regularizer_scale' : reg, 'neg_rate' : nr, 'neg_rel_rate': nrr}
            logger.info("\nTest:%d, PARAMS: %s", count, str(params))
            count = count + 1
            cur_fscore, cur_prec_at_1 = self._test_werl(dataset, columns, params)
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