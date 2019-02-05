import config
import pandas as pd
import unittest

from common import get_logger, log_quality_results
from data.cora import Cora
from data.febrl import FEBRL
from data.census import Census
from ER.tensor_transh import Tensor_TransH
from scipy import spatial

class TestTransH(unittest.TestCase):

    def _test_transh(self, dataset, threshold=config.TRANSH_THRESHOLD_CORA):
        model = dataset()
        logger = get_logger('TestTransH.' + str(model))
        entity, relation, triples, entity_pairs, true_pairs = model.get_er_model()
        transh = Tensor_TransH(entity, relation, triples, dimension=80)
        loss = transh.train()
        logger.info("Training Complete with loss: %f", loss)

        ent_embeddings = transh.get_ent_embeddings()
        alligned_pairs = []
        for i in range(0, len(entity_pairs)):
            distance = abs(spatial.distance.cosine(
                                ent_embeddings[entity_pairs[i][0]],
                                ent_embeddings[entity_pairs[i][1]]))
            if distance <= threshold:
                alligned_pairs.append(entity_pairs[i])
            #logger.info("i: %d, distance: %f true_pairs: %s", i, distance, entity_pairs[i] in true_pairs)

        result = pd.MultiIndex.from_tuples(alligned_pairs)
        log_quality_results(logger, result, true_pairs, len(entity_pairs))
        transh.close_tf_session()

    def test_transh_cora(self):
        self._test_transh(Cora, 0.25)

    def test_transh_febrl(self):
        self._test_transh(FEBRL, 0.25)

    def test_transh_census(self):
        self._test_transh(Census, 0.25)