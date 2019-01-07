import config
import pandas as pd
import unittest

from common import get_logger, log_quality_results
from data.cora import Cora
from data.febrl import FEBRL
from data.census import Census
from ER.transe import TransE
from ER.transe2 import TransE2, distanceL1

logger = get_logger('TestTransE')

class TestTransE(unittest.TestCase):

    @unittest.skip("not working")
    def test_cora2(self):
        cora = Cora()
        entity, relation, triples, entity_pairs, true_pairs = cora.get_er_model()
        transe = TransE2(entity, relation, triples, margin=1, dim = 100)
        transe.initialize()
        transe.transE(15000)

        result = []
        for (head, tail) in entity_pairs:
            sim = distanceL1(transe.entityList[head], transe.entityList[tail], transe.relationList[len(relation) - 1])
            if (head,tail) in true_pairs:
                logger.info("Head %d, Tail, %d, Similarity: %f, True link: %s", head, tail, sim, (head,tail) in true_pairs)
            if sim > config.TRANSE_THRESHOLD:
                result.append((head, tail))
                logger.info("Head %d, Tail, %d, Similarity: %f, True link: %s", head, tail, sim, (head,tail) in true_pairs)

        result = pd.MultiIndex.from_tuples(result)
        log_quality_results(logger, result, true_pairs, len(entity_pairs))

    def test_cora(self):
        cora = Cora()
        entity, relation, triples, entity_pairs, true_pairs = cora.get_er_model()
        transe = TransE(entity, relation, triples, norm='L2', dimension=50, max_epochs=50)

        result = []
        for (head, tail) in entity_pairs:
            sim = transe.similarity(head, tail)
            """
            #Debug similarity for true pairs
            if (head,tail) in true_pairs:
                logger.info("Head %d, Tail, %d, Similarity: %f, True link: %s", head, tail, sim, (head,tail) in true_pairs)
            """
            if sim == 0.5:#>= config.TRANSE_THRESHOLD:
                result.append((head, tail))
                #logger.info("Head %d, Tail, %d, Similarity: %f, True link: %s", head, tail, sim, (head,tail) in true_pairs)

        result = pd.MultiIndex.from_tuples(result)
        log_quality_results(logger, result, true_pairs, len(entity_pairs))

    @unittest.skip("not working")
    def test_febrl(self):
        febrl = FEBRL()
        entity, relation, triples, entity_pairs, true_pairs = febrl.get_er_model()
        transe = TransE(entity, relation, triples)

        result = []
        for (head, tail) in entity_pairs:
            sim = transe.similarity(head, tail, len(relation) - 1)
            if sim > config.TRANSE_THRESHOLD:
                result.append((head, tail))

        result = pd.MultiIndex.from_tuples(result)
        log_quality_results(logger, result, true_pairs, len(entity_pairs))