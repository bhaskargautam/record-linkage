'''
    Record Linkage Testing Script using Logistic Regression Method over Graph Embeddings generated using TransH
'''
import numpy as np
import pandas as pd
import re
import recordlinkage
import unittest
import xml.etree.ElementTree

from common import get_logger, log_quality_results
from data.cora import Cora
from data.febrl import FEBRL
from data.census import Census

from ER.transh import TransH

logger = get_logger('TestLogisticTransH')

from recordlinkage.base import BaseCompareFeature

class CompareEmbeddings(BaseCompareFeature):

    def _compute_vectorized(self, s1, s2):
        """Compare embeddings.

        If the embeddings in both records are identical, the similarity
        is 0. Otherwise, the similarity is difference of the two.
        """
        #logger.info("SS1: %s SS2: %s", str(s1), str(s2))
        return (abs(s1-s2) >= 1).astype(float)


class TestLogisticTransH(unittest.TestCase):

    def _test_logistic_transh(self, dataset, params):
        model = dataset()
        logger = get_logger('TestLogisticTransH.' + str(model))
        entity, relation, triples, entity_pairs, true_pairs = model.get_er_model()
        transh = TransH(entity, relation, triples,dimension=params['dimension'],
                        learning_rate=params['learning_rate'],
                        margin=params['margin'],
                        regularizer_scale=params['regularizer_scale'],
                        batchSize=params['batchSize'])
        loss = transh.train(max_epochs=params['epochs'])
        logger.info("Training Complete with loss: %f", loss)

        ent_embeddings = transh.get_ent_embeddings()
        trainDataA = pd.DataFrame(data=ent_embeddings)
        trainDataB = pd.DataFrame(data=ent_embeddings)

        compare_cl = recordlinkage.Compare()

        for i in range(0, params['dimension']):
            compare_cl.add(CompareEmbeddings(i, i, label=str(i)))

        candidate_links = pd.MultiIndex.from_tuples(entity_pairs)
        features = compare_cl.compute(candidate_links, trainDataA, trainDataB)
        logger.info("Features %s", str(features.describe()))

        logrg = recordlinkage.LogisticRegressionClassifier()
        true_links = pd.MultiIndex.from_tuples(true_pairs)
        logrg.fit(features, true_pairs)

        result = logrg.predict(features)
        log_quality_results(logger, result, true_pairs, len(entity_pairs))

    def get_default_params(self):
        return {'learning_rate': 0.1, 'margin': 1, 'dimension': 80, 'epochs': 500,
                'regularizer_scale' : 0.1, 'batchSize' : 100}

    def test_cora(self):
        self._test_logistic_transh(Cora, self.get_default_params())

    def test_febrl(self):
        self._test_logistic_transh(FEBRL, self.get_default_params())

    def test_census(self):
        self._test_logistic_transh(Census, self.get_default_params())