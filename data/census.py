import config
import numpy as np
import pandas as pd
import random
import recordlinkage
import re

from common import get_logger
from data.base_census import (
    load_census,
    CensusLocation,
    census_field_map,
    CensusFields)

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

    def __new__(cls, census_location=CensusLocation.SANT_FELIU,
                    train_years = [(1889, 1906), (1930, 1936)],
                    test_years = [(1910, 1915), (1924, 1930)],
                    validation_years = [(1906, 1910), (1936, 1940)],
                    *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Census, cls).__new__(
                                cls, *args, **kwargs)
            cls._instance.init(census_location, train_years, test_years, \
                validation_years, *args, **kwargs)
        return cls._instance

    def init(self, census_location, train_years, test_years, validation_years):
        """
            Initializer. Create separate dataframes for testing, training and validation.
            :param census_location: CensusLocation Enum
            :param train_years: List of pairs of years [(year_a, year_b)..]
            :param test_years: List of pairs of years [(year_a, year_b)..]
            :param validation_years: List of pairs of years [(year_a, year_b)..]
        """
        self.census_location = census_location
        self.field_map = census_field_map[self.census_location]

        #collect all years
        years = list(np.array(train_years).flatten())
        years.extend(list(np.array(test_years).flatten()))
        years.extend(list(np.array(validation_years).flatten()))
        years = list(set(years))
        logger.info("Reading Census Records for years: %s .....", str(years))

        data = load_census(census_location, years=years, filters=None, fields=None)
        year_field = census_field_map[census_location][CensusFields.CENSUS_YEAR]
        logger.info("Shape of Census data: %s", str(data.shape))
        logger.info("Sample Record: %s", str(data.loc[0]))

        logger.info("Available Census Years %s", str(data[year_field].unique()))
        self.trainDataA = data[data[year_field].isin([y_a for (y_a, y_b) in train_years])]
        self.trainDataB = data[data[year_field].isin([y_b for (y_a, y_b) in train_years])]
        self.testDataA = data[data[year_field].isin([y_a for (y_a, y_b) in test_years])]
        self.testDataB = data[data[year_field].isin([y_b for (y_a, y_b) in test_years])]
        self.valDataA = data[data[year_field].isin([y_a for (y_a, y_b) in validation_years])]
        self.valDataB = data[data[year_field].isin([y_b for (y_a, y_b) in validation_years])]
        logger.info("Train Population: Dataset A: %s Dataset B %s ", str(self.trainDataA.shape), str(self.trainDataB.shape))
        logger.info("Test Population: Dataset A %s Dataset B %s ", str(self.testDataA.shape), str(self.testDataB.shape))
        logger.info("Validation Population: Dataset A %s Dataset B %s ", str(self.valDataA.shape), str(self.valDataB.shape))

        #Extract training candidate pairs
        indexer = recordlinkage.Index()
        surname_field = census_field_map[census_location][CensusFields.SURNAME_2]
        indexer.block(surname_field)
        self.candidate_links = indexer.index(self.trainDataA, self.trainDataB)
        logger.info("No. of Candidate Pairs %d", (len(self.candidate_links)))

        #Extarct Training true links
        a = self.trainDataA[self.trainDataA.DNI != '']
        b = self.trainDataB[self.trainDataB.DNI != '']
        indexer = recordlinkage.Index()
        dni_field = census_field_map[census_location][CensusFields.DNI]
        indexer.block(dni_field)
        self.true_links = indexer.index(a, b)
        logger.info("Number of ALL true links: %d", len(self.true_links))
        self.true_links = [(a,b) for (a,b) in self.true_links if (a,b) in self.candidate_links]
        logger.info("Number of true links in Candidate List : %d", len(self.true_links))
        self.true_links = pd.MultiIndex.from_tuples(self.true_links)

        #Extract test candidate pairs
        indexer = recordlinkage.Index()
        indexer.block(surname_field)
        self.test_links = indexer.index(self.testDataA, self.testDataB)
        logger.info("No. of Test Pairs %d", (len(self.test_links)))

        #Extarct test true links
        a = self.testDataA[self.testDataA.DNI != '']
        b = self.testDataB[self.testDataB.DNI != '']
        indexer = recordlinkage.Index()
        dni_field = census_field_map[census_location][CensusFields.DNI]
        indexer.block(dni_field)
        self.true_test_links = indexer.index(a, b)
        logger.info("Number of ALL test links: %d", len(self.true_test_links))
        self.true_test_links = [(a,b) for (a,b) in self.true_test_links if (a,b) in self.test_links]
        logger.info("Number of test links in TEST Candidate List : %d", len(self.true_test_links))
        self.true_test_links = pd.MultiIndex.from_tuples(self.true_test_links)

        #Extract val candidate pairs
        indexer = recordlinkage.Index()
        indexer.block(surname_field)
        self.val_links = indexer.index(self.valDataA, self.valDataB)
        logger.info("No. of Validation Pairs %d", (len(self.val_links)))

        #Extarct val true links
        a = self.valDataA[self.valDataA.DNI != '']
        b = self.valDataB[self.valDataB.DNI != '']
        indexer = recordlinkage.Index()
        dni_field = census_field_map[census_location][CensusFields.DNI]
        indexer.block(dni_field)
        self.true_val_links = indexer.index(a, b)
        logger.info("Number of ALL true links: %d", len(self.true_val_links))
        self.true_val_links = [(a,b) for (a,b) in self.true_val_links if (a,b) in self.val_links]
        logger.info("Number of true links in Candidate List : %d", len(self.true_val_links))
        self.true_val_links = pd.MultiIndex.from_tuples(self.true_val_links)

    def get_comparision_object(self):
        """
            Builds the Comparison Object for six fields.
            JaroWinkler Distance for Name, Surname & relation.
            Exact Match for YOB, Civil status and occupation.
            :return : compare_cl
            :rtype : recordlinkage.Compare
        """
        compare_cl = recordlinkage.Compare()

        fname = census_field_map[self.census_location][CensusFields.FIRST_NAME]
        compare_cl.string(fname, fname, method='jarowinkler', threshold=0.85, label='normalizedName')

        sname1 = census_field_map[self.census_location][CensusFields.SURNAME_1]
        compare_cl.string(sname1, sname1, method='jarowinkler', threshold=0.85, label='normalizedSurname1')

        yob = census_field_map[self.census_location][CensusFields.YOB]
        compare_cl.exact(yob, yob, label='yearOfBirth')

        civil = census_field_map[self.census_location][CensusFields.CIVIL_STATUS]
        compare_cl.exact(civil, civil, label='civilStatus')

        relation = census_field_map[self.census_location][CensusFields.RELATION]
        compare_cl.string(relation, relation, method='jarowinkler', threshold=0.85, label='normalizedRelation')

        occupation = census_field_map[self.census_location][CensusFields.OCCUPATION]
        compare_cl.exact(occupation, occupation, label='normalizedOccupation')

        return compare_cl

    def get_er_model(self, data_type='train'):
        assert data_type in ['train', 'test', 'val', 'all'], "Invalid Data Type requested. \
            Allowed values: 'train', 'test', 'val', 'all'"

        entity = []
        relation = ["same_as", "name", "surname", "surname2", "yob", "civil", "occupation"]
        triples = []
        dni_mapping = {}
        surname_mapping = {}
        data_1940 = [] # List of individual Enity Ids in dataset A
        data_1936 = [] # List of individual Enity Ids in dataset B

        data_for_graph = []
        if data_type in ['train', 'all']:
            data_for_graph.extend([(self.trainDataA, data_1940), (self.trainDataB, data_1936)])
        if data_type in ['test', 'all']:
            data_for_graph.extend([(self.testDataA, data_1940), (self.testDataB, data_1936)])
        if data_type in ['val', 'all']:
            data_for_graph.extend([(self.valDataA, data_1940), (self.valDataB, data_1936)])

        for (dataset, data) in data_for_graph:
            for _, record in dataset.iterrows():

                #new entity for each record
                field_individual_id = self.field_map[CensusFields.ID_INDIVIDUAL]
                field_dni = self.field_map[CensusFields.DNI]
                entity.append(str(record[field_individual_id]) + "_" + \
                                    str(record[field_dni]))
                individual_id = len(entity) - 1

                #entity for each household
                field_household_id = self.field_map[CensusFields.ID_HOUSEHOLD]
                if record[field_household_id] in entity:
                    household_id = entity.index(record[field_household_id])
                else:
                    entity.append(record[field_household_id])
                    household_id = len(entity) - 1

                #populate dicticnaries for DNI and Surname
                dni_mapping[individual_id] = record[field_dni]
                surname_mapping[individual_id] = record[self.field_map[CensusFields.SURNAME_2]]
                data.append(individual_id)

                #Entity for Normalized Name
                field_fname = self.field_map[CensusFields.FIRST_NAME]
                if record[field_fname] in entity:
                    name_id = entity.index(record[field_fname])
                else:
                    entity.append(record[field_fname])
                    name_id = len(entity) - 1

                #Entity for Normalized Surname
                field_sname = self.field_map[CensusFields.SURNAME_1]
                if record[field_sname] in entity:
                    surname_id = entity.index(record[field_sname])
                else:
                    entity.append(record[field_sname])
                    surname_id = len(entity) - 1

                #Entity for Normalized Surname2
                field_sname2 = self.field_map[CensusFields.SURNAME_2]
                if record[field_sname2] in entity:
                    surname2_id = entity.index(record[field_sname2])
                else:
                    entity.append(record[field_sname2])
                    surname2_id = len(entity) - 1

                #Year of Birth
                field_yob = self.field_map[CensusFields.YOB]
                if record[field_yob] and record[field_yob] in entity:
                    yob_id = entity.index(record[field_yob])
                elif record[field_yob]:
                    entity.append(record[field_yob])
                    yob_id = len(entity) - 1
                else:
                    #check DOB for year of birth
                    field_dob = self.field_map[CensusFields.DOB]
                    field_age = self.field_map[CensusFields.AGE]
                    year = re.search('1[7-9][0-9]{2}', str(record[field_dob]))
                    if year:
                        year = year.group()
                    elif record[field_age]:
                        #compute year of birth from age
                        try:
                            field_census_year = self.field_map[CensusFields.CENSUS_YEAR]
                            year = str(int(record[field_census_year]) - int(record[field_age]))
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
                field_civil_status = self.field_map[CensusFields.CIVIL_STATUS]
                if record[field_civil_status] in entity:
                    civil_id = entity.index(record[field_civil_status])
                else:
                    entity.append(record[field_civil_status])
                    civil_id = len(entity) - 1

                #Normalized relationship with head
                field_relation = self.field_map[CensusFields.RELATION]
                if record[field_relation] in relation:
                    relation_id = relation.index(record[field_relation])
                else:
                    relation.append(record[field_relation])
                    relation_id = len(relation) - 1

                #Normalized occupation
                field_occupation = self.field_map[CensusFields.OCCUPATION]
                if record[field_occupation] in entity:
                    occupation_id = entity.index(record[field_occupation])
                else:
                    entity.append(record[field_occupation])
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

    def get_ear_model(self, data_type='train'):
        assert data_type in ['train', 'test', 'val', 'all'], "Invalid Data Type requested. \
            Allowed values: 'train', 'test', 'val', 'all'"

        entity = []
        attribute = ["name", "surname", "surname2", "yob", "civil", "occupation"]
        relation = ["same_as" "not_same_as"]
        atriples = []
        rtriples = []
        attr_value = []
        dni_mapping = {}
        surname_mapping = {}
        data_1940 = []
        data_1936 = []

        data_for_graph = []
        if data_type in ['train', 'all']:
            data_for_graph.extend([(self.trainDataA, data_1940), (self.trainDataB, data_1936)])
        if data_type in ['test', 'all']:
            data_for_graph.extend([(self.testDataA, data_1940), (self.testDataB, data_1936)])
        if data_type in ['val', 'all']:
            data_for_graph.extend([(self.valDataA, data_1940), (self.valDataB, data_1936)])


        for (dataset, data) in data_for_graph:
            for _, record in dataset.iterrows():

                #new entity for each record
                field_individual_id = self.field_map[CensusFields.ID_INDIVIDUAL]
                field_dni = self.field_map[CensusFields.DNI]
                entity.append(str(record[field_individual_id]) + "_" + \
                                str(record[field_dni]))
                individual_id = len(entity) - 1

                #entity for each household
                field_household_id = self.field_map[CensusFields.ID_HOUSEHOLD]
                if record[field_household_id] in entity:
                    household_id = entity.index(record[field_household_id])
                else:
                    entity.append(record[field_household_id])
                    household_id = len(entity) - 1

                #populate dicticnaries for DNI and Surname
                dni_mapping[individual_id] = record[field_dni]
                surname_mapping[individual_id] = record[self.field_map[CensusFields.SURNAME_2]]
                data.append(individual_id)

                #Value for Normalized Name
                field_fname = self.field_map[CensusFields.FIRST_NAME]
                if record[field_fname] in attr_value:
                    name_id = attr_value.index(record[field_fname])
                else:
                    attr_value.append(record[field_fname])
                    name_id = len(attr_value) - 1

                #Value for Normalized Surname
                field_sname = self.field_map[CensusFields.SURNAME_1]
                if record[field_sname] in attr_value:
                    surname_id = attr_value.index(record[field_sname])
                else:
                    attr_value.append(record[field_sname])
                    surname_id = len(attr_value) - 1

                #Value for Normalized Surname2
                field_sname2 = self.field_map[CensusFields.SURNAME_2]
                if record[field_sname2] in attr_value:
                    surname2_id = attr_value.index(record[field_sname2])
                else:
                    attr_value.append(record[field_sname2])
                    surname2_id = len(attr_value) - 1

                #Year of Birth
                field_yob = self.field_map[CensusFields.YOB]
                if record[field_yob] and record[field_yob] in attr_value:
                    yob_id = attr_value.index(record[field_yob])
                elif record[field_yob]:
                    attr_value.append(record[field_yob])
                    yob_id = len(attr_value) - 1
                else:
                    #check DOB for year of birth
                    field_dob = self.field_map[CensusFields.DOB]
                    field_age = self.field_map[CensusFields.AGE]
                    year = re.search('1[7-9][0-9]{2}', str(record[field_dob]))
                    if year:
                        year = year.group()
                    elif record[field_age]:
                        #compute year of birth from age
                        try:
                            field_census_year = self.field_map[CensusFields.CENSUS_YEAR]
                            year = str(int(record[field_census_year]) - int(record[field_age]))
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
                field_civil_status = self.field_map[CensusFields.CIVIL_STATUS]
                if record[field_civil_status] in attr_value:
                    civil_id = attr_value.index(record[field_civil_status])
                else:
                    attr_value.append(record[field_civil_status])
                    civil_id = len(attr_value) - 1

                #Normalized relationship with head
                field_relation = self.field_map[CensusFields.RELATION]
                rel = str(record[field_relation]).replace(' ', '_')
                if rel in relation:
                    relation_id = relation.index(rel)
                else:
                    relation.append(rel)
                    relation_id = len(relation) - 1

                #Normalized occupation
                field_occupation = self.field_map[CensusFields.OCCUPATION]
                if record[field_occupation] in attr_value:
                    occupation_id = attr_value.index(record[field_occupation])
                else:
                    attr_value.append(record[field_occupation])
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

    def get_erer_model(self, data_type='train', prior_pair_ratio=0.3):
        assert data_type in ['train', 'test', 'val', 'all'], "Invalid Data Type requested. \
            Allowed values: 'train', 'test', 'val', 'all'"

        entityA = []
        entityB = []
        relationA = ["same_as", "name", "surname", "surname2", "yob", "civil", "occupation"]
        relationB = ["same_as", "name", "surname", "surname2", "yob", "civil", "occupation"]
        triplesA = []
        triplesB = []
        dni_mapping = {}
        surname_mapping = {}
        data_A = []
        data_B = []

        data_for_graph = []
        if data_type in ['train', 'all']:
            data_for_graph.extend([(self.trainDataA, data_A, entityA, relationA, triplesA),
                                    (self.trainDataB, data_B, entityB, relationB, triplesB)])
        if data_type in ['test', 'all']:
            data_for_graph.extend([(self.testDataA, data_A, entityA, relationA, triplesA),
                                   (self.testDataB, data_B, entityB, relationB, triplesB)])
        if data_type in ['val', 'all']:
            data_for_graph.extend([(self.valDataA, data_A, entityA, relationA, triplesA),
                                   (self.valDataB, data_B, entityB, relationB, triplesB)])

        for (dataset, data, entity, relation, triples) in data_for_graph:
            for record in dataset.iterrows():
                record = record[1]

                #new entity for each record
                field_individual_id = self.field_map[CensusFields.ID_INDIVIDUAL]
                field_dni = self.field_map[CensusFields.DNI]
                entity.append(str(record[field_individual_id]) + "_" + \
                                str(record[field_dni]))
                individual_id = len(entity) - 1

                #entity for each household
                field_household_id = self.field_map[CensusFields.ID_HOUSEHOLD]
                if record[field_household_id] in entity:
                    household_id = entity.index(record[field_household_id])
                else:
                    entity.append(record[field_household_id])
                    household_id = len(entity) - 1

                #populate dicticnaries for DNI and Surname
                dni_mapping[individual_id] = record[field_dni]
                surname_mapping[individual_id] = record[self.field_map[CensusFields.SURNAME_2]]
                data.append(individual_id)

                #Entity for Normalized Name
                field_fname = self.field_map[CensusFields.FIRST_NAME]
                if record[field_fname] in entity:
                    name_id = entity.index(record[field_fname])
                else:
                    entity.append(record[field_fname])
                    name_id = len(entity) - 1

                #Entity for Normalized Surname
                field_sname = self.field_map[CensusFields.SURNAME_1]
                if record[field_sname] in entity:
                    surname_id = entity.index(record[field_sname])
                else:
                    entity.append(record[field_sname])
                    surname_id = len(entity) - 1

                #Entity for Normalized Surname2
                field_sname2 = self.field_map[CensusFields.SURNAME_2]
                if record[field_sname2] in entity:
                    surname2_id = entity.index(record[field_sname2])
                else:
                    entity.append(record[field_sname2])
                    surname2_id = len(entity) - 1

                #Year of Birth
                field_yob = self.field_map[CensusFields.YOB]
                if record[field_yob] and record[field_yob] in entity:
                    yob_id = entity.index(record[field_yob])
                elif record[field_yob]:
                    entity.append(record[field_yob])
                    yob_id = len(entity) - 1
                else:
                    #check DOB for year of birth
                    field_dob = self.field_map[CensusFields.DOB]
                    field_age = self.field_map[CensusFields.AGE]
                    year = re.search('1[7-9][0-9]{2}', str(record[field_dob]))
                    if year:
                        year = year.group()
                    elif record[field_age]:
                        #compute year of birth from age
                        try:
                            field_census_year = self.field_map[CensusFields.CENSUS_YEAR]
                            year = str(int(record[field_census_year]) - int(record[field_age]))
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
                field_civil_status = self.field_map[CensusFields.CIVIL_STATUS]
                if record[field_civil_status] in entity:
                    civil_id = entity.index(record[field_civil_status])
                else:
                    entity.append(record[field_civil_status])
                    civil_id = len(entity) - 1

                #Normalized relationship with head
                field_relation = self.field_map[CensusFields.RELATION]
                if record[field_relation] in relation:
                    relation_id = relation.index(record[field_relation])
                else:
                    relation.append(record[field_relation])
                    relation_id = len(relation) - 1

                #Normalized occupation
                field_occupation = self.field_map[CensusFields.OCCUPATION]
                if record[field_occupation] in entity:
                    occupation_id = entity.index(record[field_occupation])
                else:
                    entity.append(record[field_occupation])
                    occupation_id = len(entity) - 1

                #add triples
                triples.append((individual_id, household_id, relation_id))
                triples.append((individual_id, name_id, relation.index("name")))
                triples.append((individual_id, surname_id, relation.index("surname")))
                triples.append((individual_id, surname2_id, relation.index("surname2")))
                triples.append((individual_id, yob_id, relation.index("yob")))
                triples.append((individual_id, civil_id, relation.index("civil")))
                triples.append((individual_id, occupation_id, relation.index("occupation")))

        logger.info("Number of entitiesA: %d", len(entityA))
        logger.info("Number of relationsA: %d", len(relationA))
        logger.info("Number of TriplesA: %d", len(triplesA))
        logger.info("Number of entitiesB: %d", len(entityB))
        logger.info("Number of relationsB: %d", len(relationB))
        logger.info("Number of TriplesB: %d", len(triplesB))

        #Extract candidate pairs and true pairs
        entity_pairs = []
        true_pairs = []
        for e1 in data_A:
            for e2 in data_B:
                if surname_mapping[e1] == surname_mapping[e2]:
                    entity_pairs.append((e1, e2))
                    if dni_mapping[e1] == dni_mapping[e2] and dni_mapping[e2]:
                        true_pairs.append((e1,e2))

        logger.info("Number of ALL true pairs: %d", len(true_pairs))

        prior_pairs = []
        for i in xrange(int(len(true_pairs)*prior_pair_ratio)):
            prior_pairs.append(true_pairs.pop(random.choice(xrange(len(true_pairs)))))


        logger.info("Number of prior pairs: %d", len(prior_pairs))
        logger.info("Number of entity pairs: %d", len(entity_pairs))
        logger.info("Number of true pairs: %d", len(true_pairs))

        true_pairs = pd.MultiIndex.from_tuples(true_pairs)
        return (entityA, entityB, relationA, relationB, triplesA, triplesB, entity_pairs, prior_pairs, true_pairs)

    def get_veg_model(self, data_type='train'):
        assert data_type in ['train', 'test', 'val', 'all'], "Invalid Data Type requested. \
            Allowed values: 'train', 'test', 'val', 'all'"

        relation_value_map = {"same_as" : [], "name" : [], "surname" : [], "surname2" : [],
                                "yob" : [],  "civil" : [], "occupation" : [], "relation" : []}
        relation = ["same_as", "name", "surname", "surname2", "yob", "civil", "occupation", "relation"]
        train_triples = []
        val_triples = []
        test_triples = []

        for (true_links, datasetA, datasetB, triples) in [
            (self.true_links, self.trainDataA, self.trainDataB, train_triples),
            (self.true_val_links, self.valDataA, self.valDataB, val_triples),
            (self.true_test_links, self.testDataA, self.testDataB, test_triples)]:

            for (index_a, index_b) in true_links:

                #Load rows in consideration
                row_a = datasetA.loc[index_a]
                row_b = datasetB.loc[index_b]

                #Link Normalised First Names
                field_fname = self.field_map[CensusFields.FIRST_NAME]
                values = relation_value_map["name"]
                if row_a[field_fname] in values:
                    val_index_a = values.index(row_a[field_fname])
                else:
                    values.append(row_a[field_fname])
                    val_index_a = len(values) - 1

                if row_b[field_fname] in values:
                    val_index_b = values.index(row_b[field_fname])
                else:
                    values.append(row_b[field_fname])
                    val_index_b = len(values) - 1

                if val_index_a != val_index_b:
                    triples.append((val_index_a, val_index_b, relation.index("name")))

                #Link Normalized surname1
                field_surname = self.field_map[CensusFields.SURNAME_1]
                values = relation_value_map["surname"]
                if row_a[field_surname] in values:
                    val_index_a = values.index(row_a[field_surname])
                else:
                    values.append(row_a[field_surname])
                    val_index_a = len(values) - 1

                if row_b[field_surname] in values:
                    val_index_b = values.index(row_b[field_surname])
                else:
                    values.append(row_b[field_surname])
                    val_index_b = len(values) - 1

                if val_index_a != val_index_b:
                    triples.append((val_index_a, val_index_b, relation.index("surname")))

                #Link Normalized surname2
                field_surname2 = self.field_map[CensusFields.SURNAME_2]
                values = relation_value_map["surname2"]
                if row_a[field_surname2] in values:
                    val_index_a = values.index(row_a[field_surname2])
                else:
                    values.append(row_a[field_surname2])
                    val_index_a = len(values) - 1

                if row_b[field_surname2] in values:
                    val_index_b = values.index(row_b[field_surname2])
                else:
                    values.append(row_b[field_surname2])
                    val_index_b = len(values) - 1

                if val_index_a != val_index_b:
                    triples.append((val_index_a, val_index_b, relation.index("surname2")))

                #Link Year Of Birth
                field_yob = self.field_map[CensusFields.YOB]
                values = relation_value_map["yob"]
                if row_a[field_yob] and row_b[field_yob]:
                    if row_a[field_yob] in values:
                        val_index_a = values.index(row_a[field_yob])
                    else:
                        values.append(row_a[field_yob])
                        val_index_a = len(values) - 1

                    if row_b[field_yob] in values:
                        val_index_b = values.index(row_b[field_yob])
                    else:
                        values.append(row_b[field_yob])
                        val_index_b = len(values) - 1

                    if val_index_a != val_index_b:
                        triples.append((val_index_a, val_index_b, relation.index("yob")))

                #Link Civil Status
                field_civil = self.field_map[CensusFields.CIVIL_STATUS]
                values = relation_value_map["civil"]
                if row_a[field_civil] in values:
                    val_index_a = values.index(row_a[field_civil])
                else:
                    values.append(row_a[field_civil])
                    val_index_a = len(values) - 1

                if row_b[field_civil] in values:
                    val_index_b = values.index(row_b[field_civil])
                else:
                    values.append(row_b[field_civil])
                    val_index_b = len(values) - 1

                if val_index_a != val_index_b:
                    triples.append((val_index_a, val_index_b, relation.index("civil")))

                #Link Normalized occupation
                field_occupation = self.field_map[CensusFields.OCCUPATION]
                values = relation_value_map["occupation"]
                if row_a[field_occupation] in values:
                    val_index_a = values.index(row_a[field_occupation])
                else:
                    values.append(row_a[field_occupation])
                    val_index_a = len(values) - 1

                if row_b[field_occupation] in values:
                    val_index_b = values.index(row_b[field_occupation])
                else:
                    values.append(row_b[field_occupation])
                    val_index_b = len(values) - 1

                if val_index_a != val_index_b:
                    triples.append((val_index_a, val_index_b, relation.index("occupation")))

                #Link Normalized relationship with family head
                field_relation = self.field_map[CensusFields.RELATION]
                values = relation_value_map["relation"]
                if row_a[field_relation] in values:
                    val_index_a = values.index(row_a[field_relation])
                else:
                    values.append(row_a[field_relation])
                    val_index_a = len(values) - 1

                if row_b[field_relation] in values:
                    val_index_b = values.index(row_b[field_relation])
                else:
                    values.append(row_b[field_relation])
                    val_index_b = len(values) - 1

                if val_index_a != val_index_b:
                    triples.append((val_index_a, val_index_b, relation.index("relation")))

        logger.info("Number of Values: %s", str([len(relation_value_map[r]) for r in relation]))
        logger.info("All relations: %s", str(relation))
        logger.info("Number of Train Triples: %d", len(train_triples))
        logger.info("Number of Val Triples: %d", len(val_triples))
        logger.info("Number of Test Triples: %d", len(test_triples))

        return (relation_value_map, relation, train_triples, val_triples, test_triples)

    def get_entity_information(self, entity_name):
        """
            Returns the record row from the base dataset.
            ID_INDIVIDUAL is considered as the record_id for Census Dataset.
            :param entity_name: Record Identifer of form .+_<record_id>
            :return str: tab separated info about individual
        """
        try:
            ent_id = entity_name.split('_')[0]
        except Exception as e:
            logger.error(e)
            logger.error("Failed to get entity id for %s", str(entity_name))
            return None

        field_individual_id = self.field_map[CensusFields.ID_INDIVIDUAL]
        assert field_individual_id is not None, "Individual ID Field is None."

        for dataset in [self.trainDataA, self.trainDataB,
                        self.valDataA, self.valDataB,
                        self.testDataA, self.testDataB]:
            e = dataset[dataset[field_individual_id] == int(ent_id)]
            if len(e):
                record = e.iloc[0]
                return "\t".join([str(unicode(field).encode('utf-8', 'ignore')) for field in record])

        return None

    def get_entity_names(self, dataset):
        return ["_".join([str(dataset.iloc[i][self.field_map[CensusFields.ID_INDIVIDUAL]]),
                            str(dataset.iloc[i][self.field_map[CensusFields.DNI]])])
                            for i in range(len(dataset))]

    def get_weight_header(self):
        return  ["name", "surname1", "yob", "civil", "relation", "occupation", "probability"]

    def __str__(self):
        return config.CENSUS_FILE_PREFIX