import config
import pandas as pd
import unittest

from common import get_logger, log_quality_results, sigmoid
from data.cora import Cora
from data.febrl import FEBRL
from data.census import Census
from EAR.kr_ear import KR_EAR
from scipy import spatial


logger = get_logger('KR_EAR')

class Test_KR_EAR(unittest.TestCase):

    def _test_kr_ear(self, dataset, threshold = config.TRANSE_THRESHOLD_CORA):
        model = dataset()
        logger = get_logger('Test_KR_EAR.' + str(model))

        entity, attribute, relation, value, atriples, rtriples, entity_pairs, true_pairs = model.get_ear_model()
        kr_ear = KR_EAR(entity, attribute, relation, value, atriples, rtriples, dimension=80)
        loss = kr_ear.train()
        logger.info("Training Complete with loss: %f", loss)

        ent_embeddings = kr_ear.get_ent_embeddings()
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
        kr_ear.close_tf_session()

    def test_krear_cora(self):
        self._test_kr_ear(Cora, 0)

    def test_krear_febrl(self):
        self._test_kr_ear(FEBRL, 0.75)

    def test_krear_census(self):
        self._test_kr_ear(Census, 0.75)