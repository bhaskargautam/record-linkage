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

    def get_er_model(self):
        entity = []
        relation = ['name', 'surname', 'state', 'dob', 'postcode', 'aligned_pairs']
        triples = []

        #Build Knowledge Graph
        dataA = {}
        dataB = {}
        given_name_dict = {}
        for (dataset, data) in [(self.trainDataA, dataA), (self.trainDataB, dataB)]:
            #Also include test data (self.testDataA, dataA), (self.testDataB, dataB)]:

            for record in dataset.iterrows():
                #new entity for each record
                entity.append(record[0])
                entity_id = len(entity) - 1
                data[str(record[0])] = entity_id
                given_name_dict[str(record[0])] = record[1]['given_name']

                if record[1]['given_name'] in entity:
                    name_id = entity.index(record[1]['given_name'])
                else:
                    entity.append(record[1]['given_name'])
                    name_id = len(entity) - 1

                if record[1]['surname'] in entity:
                    surname_id = entity.index(record[1]['surname'])
                else:
                    entity.append(record[1]['surname'])
                    surname_id = len(entity) - 1

                if record[1]['state'] in entity:
                    state_id = entity.index(record[1]['state'])
                else:
                    entity.append(record[1]['state'])
                    state_id = len(entity) - 1

                if record[1]['date_of_birth'] in entity:
                    dob_id = entity.index(record[1]['date_of_birth'])
                else:
                    entity.append(record[1]['date_of_birth'])
                    dob_id = len(entity) - 1

                if record[1]['postcode'] in entity:
                    postcode_id = entity.index(record[1]['postcode'])
                else:
                    entity.append(record[1]['postcode'])
                    postcode_id = len(entity) - 1

                triples.append((entity_id, name_id, 0))
                triples.append((entity_id, surname_id, 1))
                triples.append((entity_id, state_id, 2))
                triples.append((entity_id, dob_id, 3))
                triples.append((entity_id, postcode_id, 4))

        logger.info("Number of entities: %d", len(entity))
        logger.info("All relations: %s", str(relation))
        logger.info("Number of Triples: %d", len(triples))

        entity_pairs = []
        true_pairs = []
        for a in dataA:
            for b in dataB:
                if given_name_dict[a] == given_name_dict[b]:
                    entity_pairs.append((dataA[a], dataB[b]))
                    if a.split('-')[1] == b.split('-')[1]:
                        true_pairs.append((dataA[a], dataB[b]))

        logger.info("Number of entity pairs: %d", len(entity_pairs))
        logger.info("Number of true pairs: %d", len(true_pairs))

        true_pairs = pd.MultiIndex.from_tuples(true_pairs)
        return (entity, relation, triples, entity_pairs, true_pairs)

    def get_ear_model(self):
        entity = []
        attribute = ['name', 'surname', 'state', 'dob', 'postcode']
        attr_value = []
        relation = ['aligned_pairs']
        atriples = []
        rtriples = []

        dataA = {}
        dataB = {}
        given_name_dict = {}
        for (dataset, data) in [(self.trainDataA, dataA), (self.trainDataB, dataB)]:
            #Also include test data (self.testDataA, dataA), (self.testDataB, dataB)]:

            for record in dataset.iterrows():
                #new entity for each record
                entity.append(record[0])
                entity_id = len(entity) - 1
                data[str(record[0])] = entity_id
                given_name_dict[str(record[0])] = record[1]['given_name']

                if record[1]['given_name'] in attr_value:
                    name_id = attr_value.index(record[1]['given_name'])
                else:
                    attr_value.append(record[1]['given_name'])
                    name_id = len(attr_value) - 1

                if record[1]['surname'] in attr_value:
                    surname_id = attr_value.index(record[1]['surname'])
                else:
                    attr_value.append(record[1]['surname'])
                    surname_id = len(attr_value) - 1

                if record[1]['state'] in attr_value:
                    state_id = attr_value.index(record[1]['state'])
                else:
                    attr_value.append(record[1]['state'])
                    state_id = len(attr_value) - 1

                if record[1]['date_of_birth'] in attr_value:
                    dob_id = attr_value.index(record[1]['date_of_birth'])
                else:
                    attr_value.append(record[1]['date_of_birth'])
                    dob_id = len(attr_value) - 1

                if record[1]['postcode'] in attr_value:
                    postcode_id = attr_value.index(record[1]['postcode'])
                else:
                    attr_value.append(record[1]['postcode'])
                    postcode_id = len(attr_value) - 1

                atriples.append((entity_id, name_id, 0))
                atriples.append((entity_id, surname_id, 1))
                atriples.append((entity_id, state_id, 2))
                atriples.append((entity_id, dob_id, 3))
                atriples.append((entity_id, postcode_id, 4))

        logger.info("Number of entities: %d", len(entity))
        logger.info("Number of values: %d", len(attr_value))
        logger.info("Number of attributes: %d", len(attribute))
        logger.info("Number of relations: %d", len(relation))
        logger.info("Number of Attributional Triples: %d", len(atriples))
        logger.info("Number of Relational Triples: %d", len(rtriples))

        entity_pairs = []
        true_pairs = []
        for a in dataA:
            for b in dataB:
                if given_name_dict[a] == given_name_dict[b]:
                    entity_pairs.append((dataA[a], dataB[b]))
                    if a.split('-')[1] == b.split('-')[1]:
                        true_pairs.append((dataA[a], dataB[b]))

        logger.info("Number of entity pairs: %d", len(entity_pairs))
        logger.info("Number of true pairs: %d", len(true_pairs))

        true_pairs = pd.MultiIndex.from_tuples(true_pairs)
        return (entity, attribute, relation, attr_value, atriples, rtriples, entity_pairs, true_pairs)

    def __str__(self):
        return 'FEBRL'