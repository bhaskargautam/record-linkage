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

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Census, cls).__new__(
                                cls, *args, **kwargs)
            cls._instance.init(*args, **kwargs)
        return cls._instance

    def init(self, census_location=CensusLocation.SANT_FELIU,
                    train_years = [[1940], [1936]],
                    test_years = [[1930],[1924]],
                    validation_years = [[1920], [1915]]):
        self.census_location = census_location
        self.field_map = census_field_map[self.census_location]

        #collect all years
        years = list(np.array(train_years).flatten())
        years.extend(list(np.array(test_years).flatten()))
        years.extend(list(np.array(validation_years).flatten()))
        logger.info("Reading Census Records for years: %s .....", str(years))

        data = load_census(census_location, years=years, filters=None, fields=None)
        year_field = census_field_map[census_location][CensusFields.CENSUS_YEAR]
        logger.info("Shape of Census data: %s", str(data.shape))
        logger.info("Sample Record: %s", str(data.loc[0]))

        logger.info("Available Census Years %s", str(data[year_field].unique()))
        self.trainDataA = data[data[year_field].isin(train_years[0])]
        self.trainDataB = data[data[year_field].isin(train_years[1])]
        self.testDataA = data[data[year_field].isin(test_years[0])]
        self.testDataB = data[data[year_field].isin(test_years[1])]
        self.valDataA = data[data[year_field].isin(validation_years[0])]
        self.valDataB = data[data[year_field].isin(validation_years[1])]
        logger.info("Train Population: 1940 %s 1936 %s ", str(self.trainDataA.shape), str(self.trainDataB.shape))
        logger.info("Test Population: 1930 %s 1924 %s ", str(self.testDataA.shape), str(self.testDataB.shape))
        logger.info("Validation Population: 1920 %s 1915 %s ", str(self.valDataA.shape), str(self.valDataB.shape))

        #Extract training candidate pairs
        indexer = recordlinkage.Index()
        surname_field = census_field_map[census_location][CensusFields.SURNAME_2]
        indexer.block(surname_field)
        self.candidate_links = indexer.index(self.trainDataA, self.trainDataB)
        logger.info("No. of Candidate Pairs %d", (len(self.candidate_links)))

        #Extarct Training true links (takes time...)
        true_links = []
        dni_field = census_field_map[census_location][CensusFields.DNI]
        for indexA, indexB in self.candidate_links:
            if self.trainDataA.loc[indexA][dni_field] == self.trainDataB.loc[indexB][dni_field] and\
                     self.trainDataB.loc[indexB][dni_field]:
                true_links.append((indexA, indexB))
        logger.info("Number of true links: %d", len(true_links))
        self.true_links = pd.MultiIndex.from_tuples(true_links)

        #Extract test candidate pairs
        indexer = recordlinkage.Index()
        indexer.block(surname_field)
        self.test_links = indexer.index(self.testDataA, self.testDataB)
        logger.info("No. of Test Pairs %d", (len(self.test_links)))

        #Extarct test true links (takes time...)
        true_test_links = []
        for indexA, indexB in self.test_links:
            if self.testDataA.loc[indexA][dni_field] == self.testDataB.loc[indexB][dni_field] and\
                     self.testDataB.loc[indexB][dni_field]:
                true_test_links.append((indexA, indexB))
        logger.info("Number of true test links: %d", len(true_test_links))
        self.true_test_links = pd.MultiIndex.from_tuples(true_test_links)

        #Extract val candidate pairs
        indexer = recordlinkage.Index()
        indexer.block(surname_field)
        self.val_links = indexer.index(self.valDataA, self.valDataB)
        logger.info("No. of Validation Pairs %d", (len(self.val_links)))

        #Extarct val true links (takes time...)
        true_val_links = []
        for indexA, indexB in self.val_links:
            if self.valDataA.loc[indexA][dni_field] == self.valDataB.loc[indexB][dni_field] and\
                     self.valDataB.loc[indexB][dni_field]:
                true_val_links.append((indexA, indexB))
        logger.info("Number of true Validation links: %d", len(true_val_links))
        self.true_val_links = pd.MultiIndex.from_tuples(true_val_links)

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
        relation = ["same_as"]
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

    def get_entity_information(self, entity_name):
        try:
            ent_id = entity_name.split('_')[0]
        except Exception as e:
            logger.error(e)
            logger.error("Failed to get entity id for %s", str(entity_name))
            return None

        field_individual_id = self.field_map[CensusFields.ID_INDIVIDUAL]
        assert field_individual_id is not None, "Individual ID Field is None."

        for dataset in [self.trainDataA, self.trainDataB, self.testDataA, self.testDataB]:
            e = dataset[dataset[field_individual_id] == int(ent_id)]
            if len(e):
                return e.iloc[0]
        return None


    def __str__(self):
        return config.CENSUS_FILE_PREFIX