import config
import numpy as np
import pandas as pd
import recordlinkage
import re

from common import get_logger

logger = get_logger('RL.Data.Census')

class Census(object):
    """Read Census Dataset"""
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
            cls._instance = super(Census, cls).__new__(
                                cls, *args, **kwargs)
            cls._instance.init()
        return cls._instance

    def init(self, base_file = config.CENSUS_SANT_FELIU,
                    train_years = [[1940], [1936]],
                    test_years = [[1930],[1924]],
                    validation_years = [[1920], [1915]]):
        logger.info("Reading Census Records...")
        WS = pd.read_excel(base_file, keep_default_na=False)
        data = np.array(WS)
        logger.info("Shape of Census data: %s", str(data.shape))
        logger.info("Sample Record: %s", str(data[0]))

        logger.info("Available Census Years %s", str(np.unique(data[:,4])))
        data_1940 = np.array(filter(lambda x: x[4] in train_years[0], data))
        data_1936 = np.array(filter(lambda x: x[4] in train_years[1], data))
        data_1930 = np.array(filter(lambda x: x[4] in test_years[0], data))
        data_1924 = np.array(filter(lambda x: x[4] in test_years[1], data))
        data_1920 = np.array(filter(lambda x: x[4] in validation_years[0], data))
        data_1915 = np.array(filter(lambda x: x[4] in validation_years[1], data))
        logger.info("Train Population: 1940 %s 1936 %s ", str(data_1940.shape), str(data_1936.shape))
        logger.info("Test Population: 1930 %s 1924 %s ", str(data_1930.shape), str(data_1924.shape))
        logger.info("Validation Population: 1920 %s 1915 %s ", str(data_1920.shape), str(data_1915.shape))

        self.trainDataA = pd.DataFrame(data_1940)
        self.trainDataB = pd.DataFrame(data_1936)
        self.testDataA = pd.DataFrame(data_1930)
        self.testDataB = pd.DataFrame(data_1924)
        self.valDataA = pd.DataFrame(data_1920)
        self.valDataB = pd.DataFrame(data_1915)

        #Extract training candidate pairs
        indexer = recordlinkage.Index()
        indexer.block(12)
        self.candidate_links = indexer.index(self.trainDataA, self.trainDataB)
        logger.info("No. of Candidate Pairs %d", (len(self.candidate_links)))

        #Extarct Training true links (takes time...)
        true_links = []
        for indexA, indexB in self.candidate_links:
            if data_1940[indexA][3] == data_1936[indexB][3] and data_1936[indexB][3]:
                true_links.append((indexA, indexB))
        logger.info("Number of true links: %d", len(true_links))
        self.true_links = pd.MultiIndex.from_tuples(true_links)

        #Extract test candidate pairs
        indexer = recordlinkage.Index()
        indexer.block(12)
        self.test_links = indexer.index(self.testDataA, self.testDataB)
        logger.info("No. of Test Pairs %d", (len(self.test_links)))

        #Extarct test true links (takes time...)
        true_test_links = []
        for indexA, indexB in self.test_links:
            if data_1930[indexA][3] == data_1924[indexB][3] and data_1924[indexB][3]:
                true_test_links.append((indexA, indexB))
        logger.info("Number of true test links: %d", len(true_test_links))
        self.true_test_links = pd.MultiIndex.from_tuples(true_test_links)

        #Extract val candidate pairs
        indexer = recordlinkage.Index()
        indexer.block(12)
        self.val_links = indexer.index(self.valDataA, self.valDataB)
        logger.info("No. of Validation Pairs %d", (len(self.val_links)))

        #Extarct val true links (takes time...)
        true_val_links = []
        for indexA, indexB in self.val_links:
            if data_1920[indexA][3] == data_1915[indexB][3] and data_1915[indexB][3]:
                true_val_links.append((indexA, indexB))
        logger.info("Number of true Validation links: %d", len(true_val_links))
        self.true_val_links = pd.MultiIndex.from_tuples(true_val_links)

    def get_comparision_object(self):
        compare_cl = recordlinkage.Compare()

        compare_cl.string(10, 10, method='jarowinkler', threshold=0.85, label='normalizedName')
        compare_cl.string(11, 11, method='jarowinkler', threshold=0.85, label='normalizedSurname1')
        compare_cl.exact(17, 17, label='yearOfBirth')
        compare_cl.exact(19, 19, label='civilStatus')
        compare_cl.string(21, 21, method='jarowinkler', threshold=0.85, label='normalizedRelation')
        compare_cl.exact(29, 29, label='normalizedOccupation')
        return compare_cl

    def get_er_model(self):
        entity = []
        relation = ["same_as", "name", "surname", "surname2", "yob", "civil", "occupation"]
        triples = []
        dni_mapping = {}
        surname_mapping = {}
        data_1940 = []
        data_1936 = []

        for (dataset, data) in [(self.trainDataA, data_1940), (self.trainDataB, data_1936)]:
            for record in dataset.iterrows():
                record = record[1]

                #new entity for each record
                entity.append(str(record[1]) + "_" + str(record[3]))
                individual_id = len(entity) - 1

                #entity for each household
                if record[2] in entity:
                    household_id = entity.index(record[2])
                else:
                    entity.append(record[2])
                    household_id = len(entity) - 1

                #populate dicticnaries for DNI and Surname
                dni_mapping[individual_id] = record[3]
                surname_mapping[individual_id] = record[12]
                data.append(individual_id)

                #Entity for Normalized Name
                if record[10] in entity:
                    name_id = entity.index(record[10])
                else:
                    entity.append(record[10])
                    name_id = len(entity) - 1

                #Entity for Normalized Surname
                if record[11] in entity:
                    surname_id = entity.index(record[11])
                else:
                    entity.append(record[11])
                    surname_id = len(entity) - 1

                #Entity for Normalized Surname2
                if record[12] in entity:
                    surname2_id = entity.index(record[12])
                else:
                    entity.append(record[12])
                    surname2_id = len(entity) - 1

                #Year of Birth
                if record[17] and record[17] in entity:
                    yob_id = entity.index(record[17])
                elif record[17]:
                    entity.append(record[17])
                    yob_id = len(entity) - 1
                else:
                    #check DOB for year of birth
                    year = re.search('1[7-9][0-9]{2}', str(record[16]))
                    if year:
                        year = year.group()
                    elif record[18]:
                        #compute year of birth from age
                        try:
                            year = str(int(record[4]) - int(record[18]))
                        except ValueError:
                            year = "0000"
                    else:
                        year = "0000"

                    if year in entity:
                        yob_id = entity.index(year)
                    else:
                        entity.append(year)
                        yob_id = len(entity) - 1

                #Civil Status
                if record[19] in entity:
                    civil_id = entity.index(record[19])
                else:
                    entity.append(record[19])
                    civil_id = len(entity) - 1

                #Normalized relationship with head
                if record[21] in relation:
                    relation_id = relation.index(record[21])
                else:
                    relation.append(record[21])
                    relation_id = len(relation) - 1

                #Normalized occupation
                if record[29] in entity:
                    occupation_id = entity.index(record[29])
                else:
                    entity.append(record[29])
                    occupation_id = len(entity) - 1

                #add triples
                triples.append((individual_id, household_id, relation_id))
                triples.append((individual_id, name_id, relation.index("name")))
                triples.append((individual_id, surname_id, relation.index("surname")))
                triples.append((individual_id, surname2_id, relation.index("surname2")))
                triples.append((individual_id, yob_id, relation.index("yob")))
                triples.append((individual_id, civil_id, relation.index("civil")))
                triples.append((individual_id, occupation_id, relation.index("occupation")))

        logger.info("Number of entities: %d", len(entity))
        logger.info("Number of relations: %d", len(relation))
        logger.info("Number of Triples: %d", len(triples))

        #Extract candidate pairs and true pairs
        entity_pairs = []
        true_pairs = []
        for e1 in data_1940:
            for e2 in data_1936:
                if (e1 == e2):
                    continue
                if surname_mapping[e1] == surname_mapping[e2]:
                    entity_pairs.append((e1, e2))
                    if dni_mapping[e1] == dni_mapping[e2] and dni_mapping[e2]:
                        true_pairs.append((e1,e2))

        logger.info("Number of entity pairs: %d", len(entity_pairs))
        logger.info("Number of true pairs: %d", len(true_pairs))

        true_pairs = pd.MultiIndex.from_tuples(true_pairs)
        return (entity, relation, triples, entity_pairs, true_pairs)

    def get_ear_model(self):
        entity = []
        attribute = ["name", "surname", "surname2", "yob", "civil", "occupation"]
        relation = ["same_as"]
        atriples = []
        rtriples = []
        attr_value = []
        dni_mapping = {}
        surname_mapping = {}
        data_1940 = []
        data_1936 = []

        for (dataset, data) in [(self.trainDataA, data_1940), (self.trainDataB, data_1936)]:
            for record in dataset.iterrows():
                record = record[1]

                #new entity for each record
                entity.append(str(record[1]) + "_" + str(record[3]))
                individual_id = len(entity) - 1

                #entity for each household
                if record[2] in entity:
                    household_id = entity.index(record[2])
                else:
                    entity.append(record[2])
                    household_id = len(entity) - 1

                #populate dicticnaries for DNI and Surname
                dni_mapping[individual_id] = record[3]
                surname_mapping[individual_id] = record[12]
                data.append(individual_id)

                #Value for Normalized Name
                if record[10] in attr_value:
                    name_id = attr_value.index(record[10])
                else:
                    attr_value.append(record[10])
                    name_id = len(attr_value) - 1

                #Value for Normalized Surname
                if record[11] in attr_value:
                    surname_id = attr_value.index(record[11])
                else:
                    attr_value.append(record[11])
                    surname_id = len(attr_value) - 1

                #Value for Normalized Surname2
                if record[12] in attr_value:
                    surname2_id = attr_value.index(record[12])
                else:
                    attr_value.append(record[12])
                    surname2_id = len(attr_value) - 1

                #Year of Birth
                if record[17] and record[17] in attr_value:
                    yob_id = attr_value.index(record[17])
                elif record[17]:
                    attr_value.append(record[17])
                    yob_id = len(attr_value) - 1
                else:
                    #check DOB for year of birth
                    year = re.search('1[7-9][0-9]{2}', str(record[16]))
                    if year:
                        year = year.group()
                    elif record[18]:
                        #compute year of birth from age
                        try:
                            year = str(int(record[4]) - int(record[18]))
                        except ValueError:
                            year = "0000"
                    else:
                        year = "0000"

                    if year in attr_value:
                        yob_id = attr_value.index(year)
                    else:
                        attr_value.append(year)
                        yob_id = len(attr_value) - 1

                #Civil Status
                if record[19] in attr_value:
                    civil_id = attr_value.index(record[19])
                else:
                    attr_value.append(record[19])
                    civil_id = len(attr_value) - 1

                #Normalized relationship with head
                rel = record[21].replace(' ', '_')
                if rel in relation:
                    relation_id = relation.index(rel)
                else:
                    relation.append(rel)
                    relation_id = len(relation) - 1

                #Normalized occupation
                if record[29] in attr_value:
                    occupation_id = attr_value.index(record[29])
                else:
                    attr_value.append(record[29])
                    occupation_id = len(attr_value) - 1

                #add triples
                rtriples.append((individual_id, household_id, relation_id))
                atriples.append((individual_id, name_id, 0))
                atriples.append((individual_id, surname_id, 1))
                atriples.append((individual_id, surname2_id, 2))
                atriples.append((individual_id, yob_id, 3))
                atriples.append((individual_id, civil_id, 4))
                atriples.append((individual_id, occupation_id, 5))

            logger.info("Number of entities: %d", len(entity))
            logger.info("Number of values: %d", len(attr_value))
            logger.info("Number of attributes: %d", len(attribute))
            logger.info("Number of relations: %d", len(relation))
            logger.info("Number of Attributional Triples: %d", len(atriples))
            logger.info("Number of Relational Triples: %d", len(rtriples))

        #Extract candidate pairs and true pairs
        entity_pairs = []
        true_pairs = []
        for e1 in data_1940:
            for e2 in data_1936:
                if (e1 == e2):
                    continue
                if surname_mapping[e1] == surname_mapping[e2]:
                    entity_pairs.append((e1, e2))
                    if dni_mapping[e1] == dni_mapping[e2] and dni_mapping[e2]:
                        true_pairs.append((e1,e2))

        logger.info("Number of entity pairs: %d", len(entity_pairs))
        logger.info("Number of true pairs: %d", len(true_pairs))

        true_pairs = pd.MultiIndex.from_tuples(true_pairs)
        return (entity, attribute, relation, attr_value, atriples, rtriples, entity_pairs, true_pairs)

    def get_erer_model(self):
        entityA = []
        entityB = []
        relationA = ["same_as", "name", "surname", "surname2", "yob", "civil", "occupation"]
        relationB = ["same_as", "name", "surname", "surname2", "yob", "civil", "occupation"]
        triplesA = []
        triplesB = []
        dni_mapping = {}
        surname_mapping = {}
        data_1940 = []
        data_1936 = []
        data_1930 = []
        data_1924 = []

        for (dataset, data, entity, relation, triples) in [ \
                    (self.trainDataA, data_1940, entityA, relationA, triplesA),
                    (self.trainDataB, data_1936, entityB, relationB, triplesB),
                    (self.testDataA, data_1930, entityA, relationA, triplesA),
                    (self.testDataB, data_1924, entityB, relationB, triplesB)]:

            for record in dataset.iterrows():
                record = record[1]

                #new entity for each record
                entity.append(str(record[1]) + "_" + str(record[3]))
                individual_id = len(entity) - 1

                #entity for each household
                if record[2] in entity:
                    household_id = entity.index(record[2])
                else:
                    entity.append(record[2])
                    household_id = len(entity) - 1

                #populate dicticnaries for DNI and Surname
                dni_mapping[individual_id] = record[3]
                surname_mapping[individual_id] = record[12]
                data.append(individual_id)

                #Entity for Normalized Name
                if record[10] in entity:
                    name_id = entity.index(record[10])
                else:
                    entity.append(record[10])
                    name_id = len(entity) - 1

                #Entity for Normalized Surname
                if record[11] in entity:
                    surname_id = entity.index(record[11])
                else:
                    entity.append(record[11])
                    surname_id = len(entity) - 1

                #Entity for Normalized Surname2
                if record[12] in entity:
                    surname2_id = entity.index(record[12])
                else:
                    entity.append(record[12])
                    surname2_id = len(entity) - 1

                #Year of Birth
                if record[17] and record[17] in entity:
                    yob_id = entity.index(record[17])
                elif record[17]:
                    entity.append(record[17])
                    yob_id = len(entity) - 1
                else:
                    #check DOB for year of birth
                    year = re.search('1[7-9][0-9]{2}', str(record[16]))
                    if year:
                        year = year.group()
                    elif record[18]:
                        #compute year of birth from age
                        try:
                            year = str(int(record[4]) - int(record[18]))
                        except ValueError:
                            year = "0000"
                    else:
                        year = "0000"

                    if year in entity:
                        yob_id = entity.index(year)
                    else:
                        entity.append(year)
                        yob_id = len(entity) - 1

                #Civil Status
                if record[19] in entity:
                    civil_id = entity.index(record[19])
                else:
                    entity.append(record[19])
                    civil_id = len(entity) - 1

                #Normalized relationship with head
                if record[21] in relation:
                    relation_id = relation.index(record[21])
                else:
                    relation.append(record[21])
                    relation_id = len(relation) - 1

                #Normalized occupation
                if record[29] in entity:
                    occupation_id = entity.index(record[29])
                else:
                    entity.append(record[29])
                    occupation_id = len(entity) - 1

                #add triples
                triples.append((individual_id, household_id, relation_id))
                triples.append((individual_id, name_id, relation.index("name")))
                triples.append((individual_id, surname_id, relation.index("surname")))
                triples.append((individual_id, surname2_id, relation.index("surname2")))
                triples.append((individual_id, yob_id, relation.index("yob")))
                triples.append((individual_id, civil_id, relation.index("civil")))
                triples.append((individual_id, occupation_id, relation.index("occupation")))

            logger.info("Number of entities: %d", len(entity))
            logger.info("Number of relations: %d", len(relation))
            logger.info("Number of Triples: %d", len(triples))

        #Extract candidate pairs and true pairs
        entity_pairs = []
        true_pairs = []
        for e1 in data_1930:
            for e2 in data_1924:
                if surname_mapping[e1] == surname_mapping[e2]:
                    entity_pairs.append((e1, e2))
                    if dni_mapping[e1] == dni_mapping[e2] and dni_mapping[e2]:
                        true_pairs.append((e1,e2))

        prior_pairs = []
        for e1 in data_1940:
            for e2 in data_1936:
                if dni_mapping[e1] == dni_mapping[e2] and dni_mapping[e1]:
                    prior_pairs.append((e1,e2))

        logger.info("Number of prior pairs: %d", len(prior_pairs))
        logger.info("Number of entity pairs: %d", len(entity_pairs))
        logger.info("Number of true pairs: %d", len(true_pairs))

        true_pairs = pd.MultiIndex.from_tuples(true_pairs)
        return (entityA, entityB, relationA, relationB, triplesA, triplesB, entity_pairs, prior_pairs, true_pairs)

    def get_entity_information(self, entity_name):
        try:
            ent_id = entity_name.split('_')[0]
        except Exception as e:
            logger.error(e)
            logger.error("Failed to get entity id for %s", str(entity_name))
            return None

        for dataset in [self.trainDataA, self.trainDataB, self.testDataA, self.testDataB]:
            e = [r for r in dataset.iterrows() if str(r[1][1]) == ent_id]
            if len(e):
                return e[0]
        return None


    def __str__(self):
        return config.CENSUS_FILE_PREFIX