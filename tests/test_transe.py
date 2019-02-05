import config
import pandas as pd
import unittest

from common import get_logger, log_quality_results, sigmoid
from data.cora import Cora
from data.febrl import FEBRL
from data.census import Census
from ER.transe import TransE
from scipy import spatial


logger = get_logger('TestTransE')

class TestTransE(unittest.TestCase):

    def _test_transe(self, dataset, threshold = config.TRANSE_THRESHOLD_CORA):
        model = dataset()
        logger = get_logger('TestTransE.' + str(model))

        entity, relation, triples, entity_pairs, true_pairs = model.get_er_model()
        transe = TransE(entity, relation, triples, dimension=80)
        loss = transe.train()
        logger.info("Training Complete with loss: %f", loss)

        ent_embeddings = transe.get_ent_embeddings()
        alligned_pairs = []
        for i in range(0, len(entity_pairs)):
            distance = abs(spatial.distance.cosine(
                                ent_embeddings[entity_pairs[i][0]],
                                ent_embeddings[entity_pairs[i][1]]))
            if distance <= threshold:
                alligned_pairs.append(entity_pairs[i])
                #logger.info("i: %d, distance: %f true_pairs: %s", i, distance, entity_pairs[i] in true_pairs)
        if len(alligned_pairs):
            result = pd.MultiIndex.from_tuples(alligned_pairs)
            log_quality_results(logger, result, true_pairs, len(entity_pairs))
        else:
            logger.info("No results")
        transe.close_tf_session()

    def test_transe_cora(self):
        self._test_transe(Cora, 0)

    def test_transe_febrl(self):
        self._test_transe(FEBRL, 0.3)

    def test_transe_census(self):
        self._test_transe(Census, 0.3)