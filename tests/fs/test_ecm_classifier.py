'''
    Record Linkage Testing Script for CORA dataset using ECM Classifier method.
'''
import numpy as np
import pandas as pd
import re
import recordlinkage
import unittest
import xml.etree.ElementTree

from common import get_logger, log_quality_results, InformationRetrievalMetrics
from data.cora import Cora
from data.febrl import FEBRL
from data.census import Census

class TestECMClassifier(unittest.TestCase):

    def test_cora(self):
        logger = get_logger('RL.Test.ECMClassifier.CORA')

        #Read Train data in dataset A & B
        cora = Cora()

        ## Extarct Features
        compare_cl = cora.get_comparision_object()
        features = compare_cl.compute(cora.candidate_links, cora.trainDataA, cora.trainDataB)
        logger.info("Train Features %s", str(features.describe()))

        # Train ECM Classifier
        logrg = recordlinkage.ECMClassifier()
        logrg.fit(features)

        result = logrg.predict(features)
        log_quality_results(logger, result, cora.true_links, len(cora.candidate_links))

        #validate the classifier
        compare_cl = cora.get_comparision_object()
        features = compare_cl.compute(cora.val_links, cora.valDataA, cora.valDataB)
        logger.info("Validation Features %s", str(features.describe()))

        result = logrg.predict(features)
        log_quality_results(logger, result, cora.true_val_links, len(cora.val_links))


        #Test the classifier
        compare_cl = cora.get_comparision_object()
        features = compare_cl.compute(cora.test_links, cora.testDataA, cora.testDataB)
        logger.info("Test Features %s", str(features.describe()))

        result = logrg.predict(features)
        log_quality_results(logger, result, cora.true_test_links, len(cora.test_links))

        #Log IR Stats: MRR, MAP, MP@K
        prob_series = logrg.prob(features)
        prob = [(1 - p) for p in prob_series.tolist()]
        result_prob = [(cora.test_links[i][0], cora.test_links[i][1], prob[i]) for i in range(0, len(prob))]
        ir_metrics = InformationRetrievalMetrics(result_prob, cora.true_test_links)
        ir_metrics.log_metrics(logger)

    def test_febrl(self):
        logger = get_logger('RL.Test.ECMClassifier.FEBRL')

        febrl = FEBRL()

        compare_cl = febrl.get_comparision_object()
        features = compare_cl.compute(febrl.candidate_links, febrl.trainDataA, febrl.trainDataB)
        logger.info("Train Features %s", str(features.describe()))

        # Train ECM Classifier
        logrg = recordlinkage.ECMClassifier()
        logrg.fit(features)

        result = logrg.predict(features)
        log_quality_results(logger, result, febrl.true_links, len(febrl.candidate_links))

        #Validate the classifier
        compare_cl = febrl.get_comparision_object()
        features = compare_cl.compute(febrl.val_links, febrl.valDataA, febrl.valDataB)
        logger.info("Validation Features %s", str(features.describe()))

        result = logrg.predict(features)
        log_quality_results(logger, result, febrl.true_val_links, len(febrl.val_links))

        #Test the classifier
        compare_cl = febrl.get_comparision_object()
        features = compare_cl.compute(febrl.test_links, febrl.testDataA, febrl.testDataB)
        logger.info("Test Features %s", str(features.describe()))

        result = logrg.predict(features)
        log_quality_results(logger, result, febrl.true_test_links, len(febrl.test_links))

        #Log IR Stats: MRR, MAP, MP@K
        prob_series = logrg.prob(features)
        prob = [(1 - p) for p in prob_series.tolist()]
        result_prob = [(febrl.test_links[i][0], febrl.test_links[i][1], prob[i]) for i in range(0, len(prob))]
        ir_metrics = InformationRetrievalMetrics(result_prob, febrl.true_test_links)
        ir_metrics.log_metrics(logger)

    def test_census(self):
        logger = get_logger('RL.Test.ECMClassifier.Census')

        census = Census()

        compare_cl = census.get_comparision_object()
        features = compare_cl.compute(census.candidate_links, census.trainDataA, census.trainDataB)
        logger.info("Train Features %s", str(features.describe()))

        # Train ECM Classifier
        logrg = recordlinkage.ECMClassifier()
        logrg.fit(features)

        result = logrg.predict(features)
        log_quality_results(logger, result, census.true_links, len(census.candidate_links))

        #Validate the classifier
        compare_cl = census.get_comparision_object()
        features = compare_cl.compute(census.val_links, census.valDataA, census.valDataB)
        logger.info("Validation Features %s", str(features.describe()))

        result = logrg.predict(features)
        log_quality_results(logger, result, census.true_val_links, len(census.val_links))


        #Test the classifier
        compare_cl = census.get_comparision_object()
        features = compare_cl.compute(census.test_links, census.testDataA, census.testDataB)
        logger.info("Test Features %s", str(features.describe()))

        result = logrg.predict(features)
        log_quality_results(logger, result, census.true_test_links, len(census.test_links))

        #Log IR Stats: MRR, MAP, MP@K
        prob_series = logrg.prob(features)
        prob = [(1 - p) for p in prob_series.tolist()]
        result_prob = [(census.test_links[i][0], census.test_links[i][1], prob[i]) for i in range(0, len(prob))]
        ir_metrics = InformationRetrievalMetrics(result_prob, census.true_test_links)
        ir_metrics.log_metrics(logger)
