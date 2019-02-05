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

    def get_er_model(self):
        entity = []
        relation = ["name", "surname", "surname2", "yob", "civil", "occupation"]
        triples = []
        dni_mapping = {}
        surname_mapping = {}
        data_1940 = []
        data_1936 = []

        for (dataset, data) in [(self.trainDataA, data_1940), (self.trainDataB, data_1936)]:
            for record in dataset.iterrows():
                record = record[1]

                #new entity for each record
                entity.append(record[1])
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
                        year = str(int(record[4]) - int(record[18]))
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

        entity_pairs = []
        true_pairs = []
        for e1 in data_1940:
            for e2 in data_1936:
                if surname_mapping[e1] == surname_mapping[e2]:
                    entity_pairs.append((e1, e2))
                    if dni_mapping[e1] == dni_mapping[e2]:
                        true_pairs.append((e1,e2))

        logger.info("Number of entity pairs: %d", len(entity_pairs))
        logger.info("Number of true pairs: %d", len(true_pairs))

        true_pairs = pd.MultiIndex.from_tuples(true_pairs)
        return (entity, relation, triples, entity_pairs, true_pairs)

    def get_ear_model(self):
        entity = []
        attribute = []
        relation = []
        atriples = []
        rtriples = []

        return (entity, attribute, relation, atriples, rtriples)

    def __str__(self):
        return 'Census'