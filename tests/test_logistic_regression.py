'''
    Record Linkage Testing Script for CORA dataset using Logistic Regression Method.
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

logger = get_logger('TestLogisticRegression')

class TestLogisticRegression(unittest.TestCase):

    def test_cora(self):
        #Read Train data in dataset A & B
        cora = Cora()

        ## Extarct Features
        compare_cl = cora.get_comparision_object()
        features = compare_cl.compute(cora.candidate_links, cora.trainDataA, cora.trainDataB)
        logger.info("Features %s", str(features.describe()))

        # Train Logistic Regression Classifier
        logrg = recordlinkage.LogisticRegressionClassifier()
        logrg.fit(features, cora.true_links)

        result = logrg.predict(features)
        log_quality_results(logger, result, cora.true_links, len(cora.candidate_links))

        #Test the classifier
        compare_cl = cora.get_comparision_object()
        features = compare_cl.compute(cora.test_links, cora.testDataA, cora.testDataB)
        logger.info("Features %s", str(features.describe()))

        result = logrg.predict(features)
        log_quality_results(logger, result, cora.true_test_links, len(cora.test_links))

    def test_febrl(self):
        febrl = FEBRL()

        compare_cl = febrl.get_comparision_object()
        features = compare_cl.compute(febrl.candidate_links, febrl.trainDataA, febrl.trainDataB)
        logger.info("Features %s", str(features.describe()))

        # Train ECM Classifier
        logrg = recordlinkage.LogisticRegressionClassifier()
        logrg.fit(features, febrl.true_links)

        result = logrg.predict(features)
        log_quality_results(logger, result, febrl.true_links, len(febrl.candidate_links))

        #Test the classifier
        compare_cl = febrl.get_comparision_object()
        features = compare_cl.compute(febrl.test_links, febrl.testDataA, febrl.testDataB)
        logger.info("Features %s", str(features.describe()))

        result = logrg.predict(features)
        log_quality_results(logger, result, febrl.true_test_links, len(febrl.test_links))

    def test_census(self):
        census = Census()

        compare_cl = census.get_comparision_object()
        features = compare_cl.compute(census.candidate_links, census.trainDataA, census.trainDataB)
        logger.info("Features %s", str(features.describe()))

        # Train ECM Classifier
        logrg = recordlinkage.LogisticRegressionClassifier()
        logrg.fit(features, census.true_links)

        result = logrg.predict(features)
        log_quality_results(logger, result, census.true_links, len(census.candidate_links))

        #Test the classifier
        compare_cl = census.get_comparision_object()
        features = compare_cl.compute(census.test_links, census.testDataA, census.testDataB)
        logger.info("Features %s", str(features.describe()))

        result = logrg.predict(features)
        log_quality_results(logger, result, census.true_test_links, len(census.test_links))