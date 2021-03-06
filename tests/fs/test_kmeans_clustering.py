'''
    Record Linkage Testing Script for CORA dataset using K-Means Cliustering.
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

class TestKmeansClustering(unittest.TestCase):

    def test_cora(self):
        logger = get_logger('RL.Test.KmeansClustering.CORA')

        #Read Train data in dataset A & B
        cora = Cora()

        ## Extarct Features
        compare_cl = cora.get_comparision_object()
        features = compare_cl.compute(cora.candidate_links, cora.trainDataA, cora.trainDataB)
        logger.info("Features %s", str(features.describe()))

        # Train K-Means Classifier
        logrg = recordlinkage.KMeansClassifier()
        logrg.fit(features)

        result = logrg.predict(features)
        log_quality_results(logger, result, cora.true_links, len(cora.candidate_links))

        #Test the classifier
        compare_cl = cora.get_comparision_object()
        features = compare_cl.compute(cora.test_links, cora.testDataA, cora.testDataB)
        logger.info("Features %s", str(features.describe()))

        result = logrg.predict(features)
        log_quality_results(logger, result, cora.true_test_links, len(cora.test_links))

    def test_febrl(self):
        logger = get_logger('RL.Test.KmeansClustering.FEBRL')

        febrl = FEBRL()

        compare_cl = febrl.get_comparision_object()
        features = compare_cl.compute(febrl.candidate_links, febrl.trainDataA, febrl.trainDataB)
        logger.info("Features %s", str(features.describe()))

        # Train K-Means Classifier
        logrg = recordlinkage.KMeansClassifier()
        logrg.fit(features)

        result = logrg.predict(features)
        log_quality_results(logger, result, febrl.true_links, len(febrl.candidate_links))

        #Test the classifier
        compare_cl = febrl.get_comparision_object()
        features = compare_cl.compute(febrl.test_links, febrl.testDataA, febrl.testDataB)
        logger.info("Features %s", str(features.describe()))

        result = logrg.predict(features)
        log_quality_results(logger, result, febrl.true_test_links, len(febrl.test_links))

    def test_census(self):
        logger = get_logger('RL.Test.KmeansClustering.CENSUS')

        census = Census()

        compare_cl = census.get_comparision_object()
        features = compare_cl.compute(census.candidate_links, census.trainDataA, census.trainDataB)
        logger.info("Features %s", str(features.describe()))

        # Train K-Means Classifier
        logrg = recordlinkage.KMeansClassifier(algorithm='full', max_iter=1000, random_state=42)
        logrg.fit(features)

        result = logrg.predict(features)
        log_quality_results(logger, result, census.true_links, len(census.candidate_links))

        #Test the classifier
        compare_cl = census.get_comparision_object()
        features = compare_cl.compute(census.test_links, census.testDataA, census.testDataB)
        logger.info("Features %s", str(features.describe()))

        result = logrg.predict(features)
        log_quality_results(logger, result, census.true_test_links, len(census.test_links))