import recordlinkage
import pandas as pd

from common import get_logger
from recordlinkage.datasets import load_febrl4

logger = get_logger('FEBRL')

class FEBRL(object):
    """Class to Read FEBRL Dataset"""
    trainDataA = None
    trainDataB = None
    testDataA = None
    testDataB = None
    true_links = None
    candidate_links = None
    test_links = None
    true_test_links = None

    def __init__(self):
        #Read data
        dfA_complete, dfB_complete, true_links_complete = load_febrl4(return_links=True)
        logger.info("Sample record: %s", str(dfA_complete[:1]))
        logger.info("Shape of datasets: A = %s  B = %s", str(dfA_complete.shape), str(dfB_complete.shape))
        logger.info("Shape of True Links: %s", str(true_links_complete.shape))

        # Split test & train dataset
        self.trainDataA, self.trainDataB = dfA_complete[:4000], dfB_complete[:4000]
        self.testDataA, self.testDataB = dfA_complete[-1000:], dfB_complete[-1000:]

        # Compute candidate links
        indexer = recordlinkage.Index()
        indexer.block('given_name')
        self.candidate_links = indexer.index(self.trainDataA, self.trainDataB)
        logger.info("Training Candidate Pairs: %d", (len(self.candidate_links)))

        #Extract True Links
        true_links_train = []
        for i in self.candidate_links:
            if i in true_links_complete:
                true_links_train.append(i)
        self.true_links = pd.MultiIndex.from_tuples(true_links_train)

        # Compute candidate links for testing
        indexer = recordlinkage.Index()
        indexer.block('given_name')
        self.test_links = indexer.index(self.testDataA, self.testDataB)
        logger.info("Testing Candidate Pairs: %d", (len(self.test_links)))

        #Extract True Links
        true_links_test = []
        for i in self.test_links:
            if i in true_links_complete:
                true_links_test.append(i)
        self.true_test_links = pd.MultiIndex.from_tuples(true_links_test)

    def get_comparision_object(self):
        compare_cl = recordlinkage.Compare()

        compare_cl.exact('given_name', 'given_name', label='given_name')
        compare_cl.string('surname', 'surname', method='jarowinkler', threshold=0.85, label='surname')
        compare_cl.exact('date_of_birth', 'date_of_birth', label='date_of_birth')
        compare_cl.exact('state', 'state', label='state')
        return compare_cl