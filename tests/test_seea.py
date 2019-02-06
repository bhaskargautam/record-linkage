import config
import pandas as pd
import unittest

from common import get_logger, log_quality_results, sigmoid
from data.cora import Cora
from data.febrl import FEBRL
from data.census import Census
from EAR.seea import SEEA
from scipy import spatial


logger = get_logger('TestSEEA')

class Test_SEEA(unittest.TestCase):

    def _test_seea(self, dataset, beta = config.SEEA_BETA, max_iter = 10):
        model = dataset()
        logger = get_logger('Test_SEEA.' + str(model))

        entity, attribute, relation, value, atriples, rtriples, entity_pairs, true_pairs = model.get_ear_model()
        seea = SEEA(entity, attribute, relation, value, atriples, rtriples, dimension=80)
        for j in range(0, max_iter):
            loss = seea.train()
            logger.info("Training Complete with loss: %f for iteration: %d", loss, j)

            ent_embeddings = seea.get_ent_embeddings()
            results = []
            for i in range(0, len(entity_pairs)):
                distance = abs(spatial.distance.cosine(
                                    ent_embeddings[entity_pairs[i][0]],
                                    ent_embeddings[entity_pairs[i][1]]))
                results.append((entity_pairs[i][0], entity_pairs[i][1], distance))

            results = sorted(results, key=(lambda x: x[2]))
            alligned_pairs = [(e1, e2) for (e1, e2, d) in results[:beta]]
            if len(alligned_pairs):
                result_pairs = pd.MultiIndex.from_tuples(alligned_pairs)
                log_quality_results(logger, result_pairs, true_pairs, len(entity_pairs))
            else:
                logger.info("No results")

            for (e1, e2, d) in results[:beta]:
                logger.info("iteration: %d, distance: %f true_pairs: %s", j, d, (e1, e2) in true_pairs)

            new_triples = [(e1, e2, len(relation) - 1) for (e1, e2) in alligned_pairs]
            seea.add_rel_triples(new_triples)
        seea.close_tf_session()

    def test_seea_cora(self):
        self._test_seea(Cora, 30)

    def test_seea_febrl(self):
        self._test_seea(FEBRL, 30)

    def test_seea_census(self):
        self._test_seea(Census, 30)