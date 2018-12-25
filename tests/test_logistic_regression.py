'''
    Record Linkage Testing Script for CORA dataset using Logistic Regression Method.
'''
import numpy as np
import pandas as pd
import re
import recordlinkage
import unittest
import xml.etree.ElementTree

from cora.cora import Cora
from common import get_logger, log_quality_results


logger = get_logger('TestLogisticRegression')

class TestKmeansClustering(unittest.TestCase):

    def test_cora(self):
        #Read Train data in dataset A & B
        cora = Cora()
        dataA = cora.trainDataA
        dataB = cora.trainDataB
        logger.info("Size of Dataset A %d and B  %d", len(dataA), len(dataB))

        #Extract all possible pairs & true links
        candidate_links = cora.candidate_links
        true_links = cora.true_links

        ## Extarct Features
        compare_cl = cora.get_comparision_object()
        features = compare_cl.compute(candidate_links, dataA, dataB)
        logger.info("Features %s", str(features.describe()))

        # Train Logistic Regression Classifier
        logrg = recordlinkage.LogisticRegressionClassifier()
        logrg.fit(features, true_links)

        result = logrg.predict(features)
        log_quality_results(logger, result, true_links, len(candidate_links))

        #Test the classifier

        #Split Test data in dataset A & B
        testDataA = cora.testDataA
        testDataB = cora.testDataB
        logger.info("Shape of dataset A %s & B %s", str(testDataA.shape), str(testDataB.shape))

        test_links = cora.test_links
        true_test_links = cora.true_test_links

        compare_cl = cora.get_comparision_object()
        features = compare_cl.compute(test_links, testDataA, testDataB)
        logger.info("Features %s", str(features.describe()))

        result = logrg.predict(features)
        log_quality_results(logger, result, true_test_links, len(test_links))