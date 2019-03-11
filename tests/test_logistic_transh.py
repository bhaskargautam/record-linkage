'''
    Record Linkage Testing Script using Logistic Regression Method over Graph Embeddings generated using TransH
'''
import numpy as np
import pandas as pd
import random
import re
import recordlinkage
import unittest
import xml.etree.ElementTree

from common import get_logger, log_quality_results, InformationRetrievalMetrics, export_embeddings, export_result_prob
from data.cora import Cora
from data.febrl import FEBRL
from data.census import Census
from ER.transh import TransH

class TestLogisticTransH(unittest.TestCase):

    def _test_logistic_transh_erer(self, dataset, params):
        model = dataset()
        logger = get_logger('RL.Test.LogisticTransH.ERER.' + str(model))
        entA, entB, relA, relB, triA, triB, entity_pairs, prior_pairs, true_pairs = model.get_erer_model()

        self.assertTrue(all([(tp in entity_pairs) for tp in true_pairs]))
        #Generate embeddings for datasetA
        transh = TransH(entA, relA, triA, prior_pairs,
                        dimension=params['dimension'],
                        learning_rate=params['learning_rate'],
                        margin=params['margin'],
                        regularizer_scale=params['regularizer_scale'],
                        batchSize=params['batchSize'])
        loss = transh.train(max_epochs=params['epochs'])
        logger.info("Training Complete with loss: %f", loss)
        ent_embeddingsA = transh.get_ent_embeddings()
        transh.close_tf_session()
        del transh

        #Generate embeddings for datasetB
        transh = TransH(entB, relB, triB, entity_pairs,
                        dimension=params['dimension'],
                        learning_rate=params['learning_rate'],
                        margin=params['margin'],
                        regularizer_scale=params['regularizer_scale'],
                        batchSize=params['batchSize'])
        loss = transh.train(max_epochs=params['epochs'])
        logger.info("Training Complete with loss: %f", loss)
        ent_embeddingsB = transh.get_ent_embeddings()
        transh.close_tf_session()

        ent_embeddingsA = [np.array(ent_embeddingsA[i]) for i in range(ent_embeddingsA.shape[0])]
        ent_embeddingsB = [np.array(ent_embeddingsB[i]) for i in range(ent_embeddingsB.shape[0])]
        trainDataA = pd.DataFrame(data=ent_embeddingsA)
        trainDataB = pd.DataFrame(data=ent_embeddingsB)

        #Define comparision Class
        compare_cl = recordlinkage.Compare()
        for i in range(0, params['dimension']):
            compare_cl.numeric(i, i, label=str(i)) #method='exp')

        #sample negative pairs
        train_pairs = []
        tuple_pp = set(map(tuple, prior_pairs))
        logger.info("Number of prior_pairs: %d", len(prior_pairs))
        for e1, e2 in prior_pairs:
            train_pairs.append((e1, e2))
            while True:
                neg_e2 = random.choice(xrange(0, len(entB)))
                if neg_e2 == e2 or (e1, neg_e2) in tuple_pp:
                    continue
                else:
                    train_pairs.append((e1, neg_e2))
                    break
        logger.info("Number of Train Pairs: %d", len(train_pairs))
        candidate_links = pd.MultiIndex.from_tuples(train_pairs)
        features = compare_cl.compute(candidate_links, trainDataA, trainDataB)
        logger.info("Train Features %s", str(features.describe()))

        #Train Logistic Regression Model
        logrg = recordlinkage.LogisticRegressionClassifier()
        candidate_links = pd.MultiIndex.from_tuples(prior_pairs)
        logrg.fit(features, candidate_links)

        #Test Classifier
        compare_cl = recordlinkage.Compare()
        for i in range(0, params['dimension']):
            compare_cl.numeric(i, i, label=str(i))
        candidate_links = pd.MultiIndex.from_tuples(entity_pairs)
        features = compare_cl.compute(candidate_links, trainDataA, trainDataB)
        logger.info("Test Features %s", str(features.describe()))
        result = logrg.predict(features)
        log_quality_results(logger, result, true_pairs, len(entity_pairs))

        prob_series = logrg.prob(features)
        prob = [(1 - p) for p in prob_series.tolist()]
        result_prob = [(entity_pairs[i][0], entity_pairs[i][1], prob[i]) for i in range(0, len(prob))]
        ir_metrics = InformationRetrievalMetrics(result_prob, true_pairs)
        ir_metrics.log_metrics(logger, params)

        #Export results
        export_embeddings('erer', str(model), 'LogTransH', entA, ent_embeddingsA)
        export_embeddings('erer', str(model), 'LogTransH', entB, ent_embeddingsB)
        export_result_prob(dataset, 'erer', str(model), 'LogTransH', entA, result_prob, true_pairs, entB)


    def _test_logistic_transh(self, dataset, params):
        """Note: Zero aligned pairs are returned, require fixation."""
        model = dataset()
        logger = get_logger('RL.Test.LogisticTransH.' + str(model))
        entity, relation, triples, entity_pairs, true_pairs = model.get_er_model()
        transh = TransH(entity, relation, triples, entity_pairs,
                        dimension=params['dimension'],
                        learning_rate=params['learning_rate'],
                        margin=params['margin'],
                        regularizer_scale=params['regularizer_scale'],
                        batchSize=params['batchSize'])
        loss = transh.train(max_epochs=params['epochs'])
        logger.info("Training Complete with loss: %f", loss)

        ent_embeddings = transh.get_ent_embeddings()
        ent_embeddings = [np.array(ent_embeddings[i]) for i in range(ent_embeddings.shape[0])]
        trainDataA = pd.DataFrame(data=ent_embeddings)
        trainDataB = pd.DataFrame(data=ent_embeddings)

        compare_cl = recordlinkage.Compare()
        for i in range(0, params['dimension']):
            compare_cl.numeric(i, i, label=str(i), method='gauss')

        candidate_links = pd.MultiIndex.from_tuples(entity_pairs)
        features = compare_cl.compute(candidate_links, trainDataA, trainDataB)
        logger.info("Features %s", str(features.describe()))

        logrg = recordlinkage.LogisticRegressionClassifier()
        logrg.fit(features, true_pairs)

        result = logrg.predict(features)
        log_quality_results(logger, result, true_pairs, len(entity_pairs))

        prob_series = logrg.prob(features)
        prob = [(1 - p) for p in prob_series.tolist()]
        result_prob = [(entity_pairs[i][0], entity_pairs[i][1], prob[i]) for i in range(0, len(prob))]
        ir_metrics = InformationRetrievalMetrics(result_prob, true_pairs)
        ir_metrics.log_metrics(logger, params)


    def get_default_params(self):
        return {'learning_rate': 0.1, 'margin': 1, 'dimension': 80, 'epochs': 100,
                'regularizer_scale' : 0.1, 'batchSize' : 100}

    def test_cora(self):
        self._test_logistic_transh(Cora, self.get_default_params())

    def test_febrl(self):
        self._test_logistic_transh(FEBRL, self.get_default_params())

    def test_census(self):
        self._test_logistic_transh(Census, self.get_default_params())

    def test_cora_erer(self):
        self._test_logistic_transh_erer(Cora, self.get_default_params())

    def test_febrl_erer(self):
        self._test_logistic_transh_erer(FEBRL, self.get_default_params())

    def test_census_erer(self):
        self._test_logistic_transh_erer(Census, self.get_default_params())