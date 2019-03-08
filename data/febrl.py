import config
import recordlinkage
import pandas as pd

from common import get_logger
from recordlinkage.datasets import load_febrl4

logger = get_logger('RL.Data.FEBRL')

class FEBRL(object):
    """Class to Read FEBRL Dataset"""
    #Train Data
    trainDataA = None
    trainDataB = None
    true_links = None
    candidate_links = None

    #Test Data
    testDataA = None
    testDataB = None
    test_links = None
    true_test_links = None

    #Validation Data
    valDataA = None
    valDataB = None
    val_links = None
    true_val_links = None

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(FEBRL, cls).__new__(
                                cls, *args, **kwargs)
            cls._instance.init()
        return cls._instance

    def init(self):
        #Read data
        dfA_complete, dfB_complete, true_links_complete = load_febrl4(return_links=True)
        logger.info("Sample record: %s", str(dfA_complete[:1]))
        logger.info("Shape of datasets: A = %s  B = %s", str(dfA_complete.shape), str(dfB_complete.shape))
        logger.info("Shape of True Links: %s", str(true_links_complete.shape))

        # Split test & train dataset
        self.testDataA = []
        self.testDataB = []
        self.trainDataA = []
        self.trainDataB = []
        self.valDataA = []
        self.valDataB = []

        for rec_id, row in dfA_complete.iterrows():
            id = int(rec_id.split('-')[1])
            if id < 3000:
                self.trainDataA.append(row)
            elif id < 4500:
                self.testDataA.append(row)
            else:
                self.valDataA.append(row)

        for rec_id, row in dfB_complete.iterrows():
            id = int(rec_id.split('-')[1])
            if id < 3000:
                self.trainDataB.append(row)
            elif id < 4500:
                self.testDataB.append(row)
            else:
                self.valDataB.append(row)

        self.trainDataA = pd.DataFrame(data=self.trainDataA)
        self.trainDataB = pd.DataFrame(data=self.trainDataB)
        self.testDataA = pd.DataFrame(data=self.testDataA)
        self.testDataB = pd.DataFrame(data=self.testDataB)
        self.valDataA = pd.DataFrame(data=self.valDataA)
        self.valDataB = pd.DataFrame(data=self.valDataB)

        logger.info("Train DataA shape: %s", str(self.trainDataA.shape))
        logger.info("Train DataB shape: %s", str(self.trainDataB.shape))
        logger.info("Test DataA shape: %s", str(self.testDataA.shape))
        logger.info("Test DataB shape: %s", str(self.testDataB.shape))
        logger.info("Val DataA shape: %s", str(self.valDataA.shape))
        logger.info("Val DataB shape: %s", str(self.valDataB.shape))

        # Compute candidate links for training
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
        logger.info("True Pairs: %d", (len(self.true_links)))

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
        logger.info("True Test Pairs: %d", (len(self.true_test_links)))

         # Compute candidate links for validation
        indexer = recordlinkage.Index()
        indexer.block('given_name')
        self.val_links = indexer.index(self.valDataA, self.valDataB)
        logger.info("Validation Candidate Pairs: %d", (len(self.val_links)))

        #Extract True Links
        true_links_val = []
        for i in self.val_links:
            if i in true_links_complete:
                true_links_val.append(i)
        self.true_val_links = pd.MultiIndex.from_tuples(true_links_val)
        logger.info("True Validation Pairs: %d", (len(self.true_val_links)))


    def get_comparision_object(self):
        compare_cl = recordlinkage.Compare()

        compare_cl.string('surname', 'surname', method='jarowinkler', threshold=0.85, label='surname')
        compare_cl.exact('date_of_birth', 'date_of_birth', label='date_of_birth')
        compare_cl.exact('state', 'state', label='state')
        return compare_cl

    def get_er_model(self, data_type='train'):
        assert data_type in ['train', 'test', 'val', 'all'], "Invalid Data Type requested. \
            Allowed values: 'train', 'test', 'val', 'all'"

        entity = []
        relation = ['name', 'surname', 'state', 'dob', 'postcode', 'same_as']
        triples = []

        #Build Knowledge Graph
        dataA = {}
        dataB = {}
        given_name_dict = {}

        data_for_graph = []
        if data_type in ['train', 'all']:
            data_for_graph.extend([(self.trainDataA, dataA), (self.trainDataB, dataB)])
        if data_type in ['test', 'all']:
            data_for_graph.extend([(self.testDataA, dataA), (self.testDataB, dataB)])
        if data_type in ['val', 'all']:
            data_for_graph.extend([(self.valDataA, dataA), (self.valDataB, dataB)])

        for (dataset, data) in data_for_graph:

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
                if (a == b):
                    continue
                if given_name_dict[a] == given_name_dict[b]:
                    entity_pairs.append((dataA[a], dataB[b]))
                    if a.split('-')[1] == b.split('-')[1]:
                        true_pairs.append((dataA[a], dataB[b]))

        logger.info("Number of entity pairs: %d", len(entity_pairs))
        logger.info("Number of true pairs: %d", len(true_pairs))

        true_pairs = pd.MultiIndex.from_tuples(true_pairs)
        return (entity, relation, triples, entity_pairs, true_pairs)

    def get_ear_model(self, data_type='train'):
        assert data_type in ['train', 'test', 'val', 'all'], "Invalid Data Type requested. \
            Allowed values: 'train', 'test', 'val', 'all'"

        entity = []
        attribute = ['name', 'surname', 'state', 'dob', 'postcode']
        attr_value = []
        relation = ['same_as']
        atriples = []
        rtriples = []

        dataA = {}
        dataB = {}
        given_name_dict = {}

        data_for_graph = []
        if data_type in ['train', 'all']:
            data_for_graph.extend([(self.trainDataA, dataA), (self.trainDataB, dataB)])
        if data_type in ['test', 'all']:
            data_for_graph.extend([(self.testDataA, dataA), (self.testDataB, dataB)])
        if data_type in ['val', 'all']:
            data_for_graph.extend([(self.valDataA, dataA), (self.valDataB, dataB)])

        for (dataset, data) in data_for_graph:
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
                if (a == b):
                    continue
                if given_name_dict[a] == given_name_dict[b]:
                    entity_pairs.append((dataA[a], dataB[b]))
                    if a.split('-')[1] == b.split('-')[1]:
                        true_pairs.append((dataA[a], dataB[b]))

        logger.info("Number of entity pairs: %d", len(entity_pairs))
        logger.info("Number of true pairs: %d", len(true_pairs))

        true_pairs = pd.MultiIndex.from_tuples(true_pairs)
        return (entity, attribute, relation, attr_value, atriples, rtriples, entity_pairs, true_pairs)

    def get_erer_model(self):
        entityA = []
        entityB = []
        relationA = ['name', 'surname', 'state', 'dob', 'postcode', 'same_as']
        relationB = ['name', 'surname', 'state', 'dob', 'postcode', 'same_as']
        triplesA = []
        triplesB = []

        #Build Knowledge Graph
        dataA = {}
        dataB = {}
        dataC = {}
        dataD = {}
        given_name_dict = {}
        for (dataset, data, entity, relation, triples) in [ \
                (self.trainDataA, dataA, entityA, relationA, triplesA),
                (self.trainDataB, dataB, entityB, relationB, triplesB),
                (self.testDataA, dataC, entityA, relationA, triplesA),
                (self.testDataB, dataD, entityB, relationB, triplesB)]:

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

        prior_pairs = []
        for a in dataA:
            for b in dataB:
                if a.split('-')[1] == b.split('-')[1]:
                    prior_pairs.append((dataA[a], dataB[b]))

        entity_pairs = []
        true_pairs = []
        for a in dataC:
            for b in dataD:
                if given_name_dict[a] == given_name_dict[b]:
                    entity_pairs.append((dataC[a], dataD[b]))
                if a.split('-')[1] == b.split('-')[1]:
                    true_pairs.append((dataC[a], dataD[b]))

        logger.info("Number of prior pairs: %d", len(prior_pairs))
        logger.info("Number of entity pairs: %d", len(entity_pairs))
        logger.info("Number of true pairs: %d", len(true_pairs))

        true_pairs = pd.MultiIndex.from_tuples(true_pairs)
        return (entityA, entityB, relationA, relationB, triplesA, triplesB, entity_pairs, prior_pairs, true_pairs)

    def get_entity_information(self, entity_name):
        for dataset in [self.trainDataA, self.trainDataB, self.testDataA, self.testDataB]:
            e = [e for e in dataset.iterrows() if e[0] == entity_name]
            if len(e):
                return e[0]
        return None

    def __str__(self):
        return config.FEBRL_FILE_PREFIX