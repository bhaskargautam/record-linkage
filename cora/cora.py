import xml.etree.ElementTree
import pandas as pd
import re
import random
from common import get_logger
import config
import recordlinkage
import random

logger = get_logger('CORA')

class Cora(object):
    """Class to read Cora Dataset"""
    trainDataA = None
    trainDataB = None
    testDataA = None
    testDataB = None
    true_links = None
    candidate_links = None
    test_links = None
    true_test_links = None

    def __init__(self):
        e = xml.etree.ElementTree.parse(config.CORA_XML_PATH).getroot()
        logger.info("Sample Record from CORA dataset: ")
        logger.info(xml.etree.ElementTree.tostring(e.find('NEWREFERENCE')))

        data = {}
        for record in e.findall('NEWREFERENCE'):
            dni = re.search('[a-z]+[0-9]+[a-z]*', record.text)
            dni = re.search('[a-z]+', record.text) if not dni else dni
            dni = dni.group() if dni else record.text

            if dni in data:
                data[dni].append(record)
            else:
                data[dni] = [record]

        dataA = []
        dataB = []
        testA = []
        testB = []
        count = 116 #Np = Number of True Links
        #Divide duplicate pairs to each dataset
        for dni in data:
            if len(data[dni]) > 1:
                if count > 0:
                    dataA.append(data[dni][0])
                    dataB.append(data[dni][1])
                    count = count - 1
                else:
                    testA.append(data[dni][0])
                    testB.append(data[dni][1])

        #Add noise enities to both dataset which are not linked.
        for dni in data:
            if len(data[dni]) == 1:
                if len(dataA) < 145: #and random.randint(0,1):
                    dataA.append(data[dni][0])
                elif len(dataB) < 143:
                    dataB.append(data[dni][0])
                elif random.randint(0, 1):
                    testA.append(data[dni][0])
                else:
                    testB.append(data[dni][0])

        logger.info("Size of Dataset A %d and B  %d", len(dataA), len(dataB))
        df_a = df_b = tdf_a = tdf_b = {  'dni' : [], 'author' : [], 'publisher' : [], 'date' : [],
                'title' : [], 'journal' : [], 'volume' : [], 'pages' : [], 'address' : []}
        for (df, dataset) in [(df_a, dataA), (df_b, dataB), (tdf_a, testA), (tdf_b, testB)]:
            for record in dataset:
                dni = re.search('[a-z]+[0-9]+[a-z]*', record.text)
                dni = re.search('[a-z]+', record.text) if not dni else dni
                df['dni'].append(dni.group() if dni else record.text)
                df['author'].append(unicode(record.find('author').text if record.find('author') is not None else '','utf-8'))
                df['title'].append(unicode(record.find('title').text if record.find('title') is not None else u''))
                df['publisher'].append(unicode(record.find('publisher').text if record.find('publisher') is not None else ''))
                df['date'].append(unicode(record.find('date').text if record.find('date') is not None else ''))
                df['journal'].append(unicode(record.find('journal').text if record.find('journal') is not None else ''))
                df['volume'].append(unicode(record.find('volume').text if record.find('volume') is not None else ''))
                df['pages'].append(unicode(record.find('pages').text if record.find('pages') is not None else ''))
                df['address'].append(unicode(record.find('address').text if record.find('address') is not None else ''))

        self.trainDataA = pd.DataFrame(data=df_a)
        self.trainDataB = pd.DataFrame(data=df_b)
        self.testDataA = pd.DataFrame(data=tdf_a)
        self.testDataB = pd.DataFrame(data=tdf_b)

        #Extract all possible pairs for training
        indexer = recordlinkage.Index()
        indexer.full()
        self.candidate_links = indexer.index(self.trainDataA, self.trainDataB)
        logger.info("No. of Candidate Pairs %d", (len(self.candidate_links)))

        #Extarct true links (takes time...)
        true_links = []
        for indexA, indexB in self.candidate_links:
            if df_a['dni'][indexA] == df_b['dni'][indexB]:
                true_links.append((indexA, indexB))
        logger.info("Number of true links: %d", len(true_links))
        self.true_links = pd.MultiIndex.from_tuples(true_links)

        #Extract all possible pairs for test
        indexer = recordlinkage.Index()
        indexer.full()
        self.test_links = indexer.index(self.testDataA, self.testDataB)
        logger.info("Number Candidate Pairs for testing: %d", (len(self.test_links)))

        #Extarct test true links (takes time...)
        true_test_links = []
        for indexA, indexB in self.test_links:
            if tdf_a['dni'][indexA] == tdf_b['dni'][indexB]:
                true_test_links.append((indexA, indexB))
        logger.info("Number of true links: %d", len(true_test_links))
        self.true_test_links = pd.MultiIndex.from_tuples(true_test_links)

    def get_comparision_object(self):
        compare_cl = recordlinkage.Compare()

        compare_cl.string('title', 'title', method='jarowinkler', threshold=0.85, label='title')
        compare_cl.string('author', 'author', method='jarowinkler', threshold=0.85, label='author')
        compare_cl.string('publisher', 'publisher', method='jarowinkler', threshold=0.85, label='publisher')
        compare_cl.string('date', 'date', method='jarowinkler', threshold=0.85, label='date')
        compare_cl.string('pages', 'pages', method='jarowinkler', threshold=0.85, label='pages')
        compare_cl.string('volume', 'volume', method='jarowinkler', threshold=0.85, label='volume')
        compare_cl.string('journal', 'journal', method='jarowinkler', threshold=0.85, label='journal')
        compare_cl.string('address', 'address', method='jarowinkler', threshold=0.85, label='address')

        return compare_cl