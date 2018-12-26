import config
import numpy as np
import pandas as pd
import recordlinkage

from common import get_logger

logger = get_logger('Census')

class Census(object):
    """Read Census Dataset"""
    trainDataA = None
    trainDataB = None
    testDataA = None
    testDataB = None
    true_links = None
    candidate_links = None
    test_links = None
    true_test_links = None

    def __init__(self):
        logger.info("Reading Census Records...")
        WS = pd.read_excel(config.CENSUS_SANT_FELIU)
        data = np.array(WS)
        logger.info("Shape of Census data: %s", str(data.shape))
        logger.info("Sample Record: %s", str(data[0]))

        logger.info("Available Census Years %s", str(np.unique(data[:,4])))
        data_1940 = np.array(filter(lambda x: x[4] == 1940, data))
        data_1936 = np.array(filter(lambda x: x[4] == 1936, data))
        data_1930 = np.array(filter(lambda x: x[4] == 1930, data))
        data_1924 = np.array(filter(lambda x: x[4] == 1924, data))
        logger.info("Population: 1940 %s 1936 %s ", str(data_1940.shape), str(data_1936.shape))
        logger.info("Population: 1930 %s 1924 %s ", str(data_1930.shape), str(data_1924.shape))

        self.trainDataA = pd.DataFrame(data_1940)
        self.trainDataB = pd.DataFrame(data_1936)
        self.testDataA = pd.DataFrame(data_1930)
        self.testDataB = pd.DataFrame(data_1924)

        indexer = recordlinkage.Index()
        indexer.block(12)
        self.candidate_links = indexer.index(self.trainDataA, self.trainDataB)
        logger.info("No. of Candidate Pairs %d", (len(self.candidate_links)))

        #Extarct true links (takes time...)
        true_links = []
        for indexA, indexB in self.candidate_links:
            if data_1940[indexA][3] == data_1936[indexB][3]:
                true_links.append((indexA, indexB))
        logger.info("Number of true links: %d", len(true_links))
        self.true_links = pd.MultiIndex.from_tuples(true_links)

        indexer = recordlinkage.Index()
        indexer.block(12)
        self.test_links = indexer.index(self.testDataA, self.testDataB)
        logger.info("No. of Test Pairs %d", (len(self.test_links)))

        #Extarct true links (takes time...)
        true_test_links = []
        for indexA, indexB in self.test_links:
            if data_1930[indexA][3] == data_1924[indexB][3]:
                true_test_links.append((indexA, indexB))
        logger.info("Number of true test links: %d", len(true_test_links))
        self.true_test_links = pd.MultiIndex.from_tuples(true_test_links)

    def get_comparision_object(self):
        compare_cl = recordlinkage.Compare()

        compare_cl.string(10, 10, method='jarowinkler', threshold=0.85, label='normalizedName')
        compare_cl.string(11, 11, method='jarowinkler', threshold=0.85, label='normalizedSurname1')
        compare_cl.string(12, 12, method='jarowinkler', threshold=0.85, label='normalizedSurname2')
        compare_cl.exact(17, 17, label='yearOfBirth')
        compare_cl.exact(19, 19, label='civilStatus')
        compare_cl.string(21, 21, method='jarowinkler', threshold=0.85, label='normalizedRelation')
        compare_cl.exact(29, 29, label='normalizedOccupation')
        return compare_cl