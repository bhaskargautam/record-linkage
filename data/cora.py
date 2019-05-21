import config
import copy
import numpy as np
import pandas as pd
import random
import re
import recordlinkage
import xml.etree.ElementTree

from common import get_logger

logger = get_logger('RL.Data.CORA')

class Cora(object):
    """Class to read Cora Dataset"""
    data = None

    #Training Data
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
            cls._instance = super(Cora, cls).__new__(
                                cls, *args, **kwargs)
            cls._instance.init()
        return cls._instance

    def init(self):
        e = xml.etree.ElementTree.parse(config.CORA_XML_PATH).getroot()
        logger.info("Sample Record from CORA dataset: ")
        logger.info(xml.etree.ElementTree.tostring(e.find('NEWREFERENCE')))

        self.data = {}
        dataA = []
        dataB = []
        testA = []
        testB = []
        valA = []
        valB =[]
        db_choice = 0
        for record in e.findall('NEWREFERENCE'):
            dni = re.search('[a-z]+[0-9]+[a-z]*', record.text)
            dni = re.search('[a-z]+', record.text) if not dni else dni
            dni = dni.group() if dni else record.text

            if dni in self.data:
                self.data[dni].append(record)
            else:
                self.data[dni] = [record]

            #db_choice = random.randint(0, 3)
            if db_choice == 0 or db_choice == 4 or db_choice == 8:
                dataA.append(record)
            elif db_choice == 1 or db_choice == 5 or db_choice == 9:
                dataB.append(record)
            elif db_choice == 2 or db_choice == 6:
                testA.append(record)
            elif db_choice == 3 or db_choice == 7:
                testB.append(record)
            elif db_choice == 10:
                valA.append(record)
            elif db_choice == 11:
                valB.append(record)
            db_choice = (db_choice + 1) % 12

        logger.info("Size of Training Dataset A %d and B  %d", len(dataA), len(dataB))
        logger.info("Size of Testing Dataset A %d and B  %d", len(testA), len(testB))
        logger.info("Size of Validation Dataset A %d and B  %d", len(valA), len(valB))

        df_a = {  'dni' : [], 'author' : [], 'publisher' : [], 'date' : [],
                'title' : [], 'journal' : [], 'volume' : [], 'pages' : [], 'address' : [], 'id' : []}
        df_b = copy.deepcopy(df_a)
        tdf_a = copy.deepcopy(df_a)
        tdf_b = copy.deepcopy(df_a)
        vdf_a = copy.deepcopy(df_a)
        vdf_b = copy.deepcopy(df_a)
        for (df, dataset) in [(df_a, dataA), (df_b, dataB), (tdf_a, testA), (tdf_b, testB), (vdf_a, valA), (vdf_b, valB)]:
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
                df['id'].append(record.get('id'))

        self.trainDataA = pd.DataFrame(data=df_a)
        self.trainDataB = pd.DataFrame(data=df_b)
        self.testDataA = pd.DataFrame(data=tdf_a)
        self.testDataB = pd.DataFrame(data=tdf_b)
        self.valDataA = pd.DataFrame(data=vdf_a)
        self.valDataB = pd.DataFrame(data=vdf_b)

        #Extract all possible pairs for training
        indexer = recordlinkage.Index()
        indexer.full()
        self.candidate_links = indexer.index(self.trainDataA, self.trainDataB)
        logger.info("No. of Candidate Pairs %d", (len(self.candidate_links)))

        #Extarct true links (takes time...)
        true_links = []
        for indexA, indexB in self.candidate_links:
            if df_a['dni'][indexA] == df_b['dni'][indexB] and df_a['dni'][indexA]:
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
        logger.info("Number of true links for testing: %d", len(true_test_links))
        self.true_test_links = pd.MultiIndex.from_tuples(true_test_links)

        #Extract all possible pairs for Validation
        indexer = recordlinkage.Index()
        indexer.full()
        self.val_links = indexer.index(self.valDataA, self.valDataB)
        logger.info("Number Candidate Pairs for validation: %d", (len(self.val_links)))

        #Extarct test true links (takes time...)
        true_val_links = []
        for indexA, indexB in self.val_links:
            if vdf_a['dni'][indexA] == vdf_b['dni'][indexB]:
                true_val_links.append((indexA, indexB))
        logger.info("Number of true links for validation: %d", len(true_val_links))
        self.true_val_links = pd.MultiIndex.from_tuples(true_val_links)

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

    def get_er_model(self, data_type='train'):
        assert data_type in ['train', 'test', 'val', 'all'], "Invalid Data Type requested. \
            Allowed values: 'train', 'test', 'val', 'all'"
        entity = []
        relation = []
        triples = []
        dni_mapping = {}
        enitity_id_mapping = {}

        allowed_record_ids_A = []
        allowed_record_ids_B = []
        if data_type in ['train', 'all']:
            allowed_record_ids_A.extend(self.trainDataA['id'].tolist())
            allowed_record_ids_B.extend(self.trainDataB['id'].tolist())
        if data_type in ['test', 'all']:
            allowed_record_ids_A.extend(self.testDataA['id'].tolist())
            allowed_record_ids_B.extend(self.testDataB['id'].tolist())
        if data_type in ['val', 'all']:
            allowed_record_ids_A.extend(self.valDataA['id'].tolist())
            allowed_record_ids_B.extend(self.valDataB['id'].tolist())

        allowed_record_ids_A = set(allowed_record_ids_A)
        allowed_record_ids_B = set(allowed_record_ids_B)

        for dni in self.data:
            for record in self.data[dni]:
                if str(record.get("id")) not in allowed_record_ids_A and \
                        str(record.get("id")) not in allowed_record_ids_B:
                    logger.debug("Skipping record id %s, since not in requested\
                                 data_type %s", str(record.get("id")), data_type)
                    continue

                #Create new entity of each record.
                entity.append("cora" + str(record.get("id") + "_" + str(dni)))
                entity_id = len(entity) - 1;
                enitity_id_mapping[str(record.get("id"))] = entity_id

                dni_mapping[str(entity_id)] = dni
                xeid = xml.etree.ElementTree.Element("entity_id")
                xeid.text = str(entity_id)
                record.insert(1, xeid)

                for rel in record._children:
                    if rel.tag in ['Pages', 'booktile', 'month', 'entity_id']:
                        continue #These Releation only appear once, so skip.

                    if rel.tag in relation:
                        relation_id = relation.index(rel.tag)
                    else:
                        relation.append(rel.tag)
                        relation_id = len(relation) - 1

                    value = None
                    if not rel.text:
                        continue
                    else:
                        value = unicode(rel.text.strip())
                        value = value.replace('(', '')
                        value = value.replace(')', '')
                        value = value.replace(';', '')
                        if rel.tag in ['date', 'year']:
                            value = value.replace('.', '')
                            value = value.replace(',', '')
                        elif rel.tag == 'pages':
                            m = re.search('[0-9\-]+', value)
                            if m:
                                value = m.group()
                            else:
                                continue
                        value = value.lower()

                    if rel.tag == 'author':
                        #KG2: separate enitity for each author
                        authors = re.split(',|and|&', value)

                        for author_name in authors:
                            author_name = author_name.strip()
                            if len(author_name) < 4:
                                continue
                            if author_name in entity:
                                author_id = entity.index(author_name)
                            else:
                                entity.append(author_name)
                                author_id = len(entity) - 1
                            triples.append((entity_id, author_id, relation_id))
                    elif value in entity:
                        value_id = entity.index(value)
                        triples.append((entity_id, value_id, relation_id))
                    else:
                        entity.append(value)
                        value_id = len(entity) - 1;
                        triples.append((entity_id, value_id, relation_id))

        #Add new relation for aligned pairs
        relation.append('same_as')
        alligned_relation_id = len(relation) - 1

        logger.info("Number of entities: %d", len(entity))
        logger.info("Number of relations: %d", len(relation))
        logger.info("Number of Triples: %d", len(triples))

        logger.info("Sample Entities: %s", str(entity[:10]))
        logger.info("All Relations: %s", str(relation))
        logger.info("Sample Triples: %s", str(triples[:10]))

        #Extract candidate links and true links
        entity_pairs = []
        true_pairs = []
        for a in allowed_record_ids_A:
            a_id = enitity_id_mapping[str(a)]
            for b in allowed_record_ids_B:
                b_id = enitity_id_mapping[str(b)]
                if (a_id == b_id):
                    continue
                entity_pairs.append((a_id,b_id))
                if dni_mapping[str(a_id)] == dni_mapping[str(b_id)]:
                    true_pairs.append((a_id,b_id))

        logger.info("Number of entity pairs: %d", len(entity_pairs))
        logger.info("Number of true pairs: %d", len(true_pairs))

        true_pairs = pd.MultiIndex.from_tuples(true_pairs)
        return (entity, relation, triples, entity_pairs, true_pairs)

    def get_ear_model(self, data_type='train'):
        assert data_type in ['train', 'test', 'val', 'all'], "Invalid Data Type requested. \
            Allowed values: 'train', 'test', 'val', 'all'"

        entity = []
        attribute = []
        relation = []
        atriples = []
        rtriples = []
        attr_value = []
        dni_mapping = {}
        enitity_id_mapping = {}

        allowed_record_ids_A = []
        allowed_record_ids_B = []
        if data_type in ['train', 'all']:
            allowed_record_ids_A.extend(self.trainDataA['id'].tolist())
            allowed_record_ids_B.extend(self.trainDataB['id'].tolist())
        if data_type in ['test', 'all']:
            allowed_record_ids_A.extend(self.testDataA['id'].tolist())
            allowed_record_ids_B.extend(self.testDataB['id'].tolist())
        if data_type in ['val', 'all']:
            allowed_record_ids_A.extend(self.valDataA['id'].tolist())
            allowed_record_ids_B.extend(self.valDataB['id'].tolist())

        allowed_record_ids_A = set(allowed_record_ids_A)
        allowed_record_ids_B = set(allowed_record_ids_B)

        for dni in self.data:
            for record in self.data[dni]:
                if str(record.get("id")) not in allowed_record_ids_A and \
                        str(record.get("id")) not in allowed_record_ids_B:
                    logger.debug("Skipping record id %s, since not in requested\
                                 data_type %s", str(record.get("id")), data_type)
                    continue

                #Create new entity for each record.
                entity.append("cora" + str(record.get("id") + "_" + str(dni)))
                entity_id = len(entity) - 1;
                dni_mapping[str(entity_id)] = dni
                enitity_id_mapping[str(record.get("id"))] = entity_id

                xeid = xml.etree.ElementTree.Element("entity_id")
                xeid.text = str(entity_id)
                record.insert(1, xeid)

                for rel in record._children:
                    if rel.tag in ['Pages', 'booktile', 'month', 'entity_id']:
                        continue #These Releation only appear once, so skip.

                    if rel.tag in attribute:
                        attribute_id = attribute.index(rel.tag)
                    else:
                        attribute.append(rel.tag)
                        attribute_id = len(attribute) - 1

                    value = None
                    if not rel.text:
                        continue
                    else:
                        value = unicode(rel.text.strip())
                        value = value.replace('(', '')
                        value = value.replace(')', '')
                        value = value.replace(';', '')
                        if rel.tag in ['date', 'year']:
                            value = value.replace('.', '')
                            value = value.replace(',', '')
                        elif rel.tag == 'pages':
                            m = re.search('[0-9\-]+', value)
                            if m:
                                value = m.group()
                            else:
                                continue
                        value = value.lower()

                    if rel.tag == 'author':
                        #KG2: separate enitity for each author
                        authors = re.split(',|and|&', value)

                        for author_name in authors:
                            author_name = author_name.strip()
                            if len(author_name) < 4:
                                continue
                            if author_name in attr_value:
                                author_id = attr_value.index(author_name)
                            else:
                                attr_value.append(author_name)
                                author_id = len(attr_value) - 1

                            atriples.append((entity_id, author_id, attribute_id))
                    else:
                        if value in attr_value:
                            value_id = attr_value.index(value)
                        else:
                            attr_value.append(value)
                            value_id = len(attr_value) - 1;

                        atriples.append((entity_id, value_id, attribute_id))

        #Add 2 new relations for aligned and non-aligned pairs
        relation.append('same_as')
        alligned_relation_id = len(relation) - 1
        relation.append('not_same_as')

        logger.info("Number of entities: %d", len(entity))
        logger.info("Number of values: %d", len(attr_value))
        logger.info("Number of attributes: %d", len(attribute))
        logger.info("Number of relations: %d", len(relation))
        logger.info("Number of Attributional Triples: %d", len(atriples))
        logger.info("Number of Relational Triples: %d", len(rtriples))

        entity_pairs = []
        true_pairs = []
        for a in allowed_record_ids_A:
            a_id = enitity_id_mapping[str(a)]
            for b in allowed_record_ids_B:
                b_id = enitity_id_mapping[str(b)]
                if (a_id == b_id):
                    continue
                entity_pairs.append((a_id,b_id))
                if dni_mapping[str(a_id)] == dni_mapping[str(b_id)]:
                    true_pairs.append((a_id,b_id))

        logger.info("Number of entity pairs: %d", len(entity_pairs))
        logger.info("Number of true pairs: %d", len(true_pairs))

        true_pairs = pd.MultiIndex.from_tuples(true_pairs)
        return (entity, attribute, relation, attr_value, atriples, rtriples, entity_pairs, true_pairs)

    def get_erer_model(self, data_type='train', prior_pair_ratio=0.3):
        assert data_type in ['train', 'test', 'val', 'all'], "Invalid Data Type requested. \
            Allowed values: 'train', 'test', 'val', 'all'"

        entityA = []
        entityB = []
        relationA = ['same_as']
        relationB = ['same_as']
        triplesA = []
        triplesB = []
        dni_mapping_A = {}
        dni_mapping_B = {}
        enitity_id_mapping = {}

        entity, relation, triples, dni_mapping = None, None, None, None
        record_ids_A = []
        record_ids_B = []
        if data_type in ['train', 'all']:
            record_ids_A.extend(self.trainDataA['id'].tolist())
            record_ids_B.extend(self.trainDataB['id'].tolist())
        elif data_type in ['test', 'all']:
            record_ids_A.extend(self.testDataA['id'].tolist())
            record_ids_B.extend(self.testDataB['id'].tolist())
        elif data_type in ['val', 'all']:
            record_ids_A.extend(self.valDataA['id'].tolist())
            record_ids_B.extend(self.valDataB['id'].tolist())

        for dni in self.data:
            for record in self.data[dni]:
                id = str(record.get('id'))
                if id in record_ids_A:
                    entity = entityA
                    relation = relationA
                    triples = triplesA
                    dni_mapping = dni_mapping_A
                elif id in record_ids_B:
                    entity = entityB
                    relation = relationB
                    triples = triplesB
                    dni_mapping = dni_mapping_B
                else:
                    continue

                entity.append("cora" + str(record.get("id") + "_" + str(dni)))
                entity_id = len(entity) - 1;
                enitity_id_mapping[str(record.get("id"))] = entity_id

                dni_mapping[str(entity_id)] = dni
                xeid = xml.etree.ElementTree.Element("entity_id")
                xeid.text = str(entity_id)
                record.insert(1, xeid)

                for rel in record._children:
                    if rel.tag in ['Pages', 'booktile', 'month', 'entity_id']:
                        continue #These Releation only appear once, so skip.

                    if rel.tag in relation:
                        relation_id = relation.index(rel.tag)
                    else:
                        relation.append(rel.tag)
                        relation_id = len(relation) - 1

                    value = None
                    if not rel.text:
                        continue
                    else:
                        value = unicode(rel.text.strip())
                        value = value.replace('(', '')
                        value = value.replace(')', '')
                        value = value.replace(';', '')
                        if rel.tag in ['date', 'year']:
                            value = value.replace('.', '')
                            value = value.replace(',', '')
                        elif rel.tag == 'pages':
                            m = re.search('[0-9\-]+', value)
                            if m:
                                value = m.group()
                            else:
                                continue
                        value = value.lower()

                    if rel.tag == 'author':
                        #KG2: separate enitity for each author
                        authors = re.split(',|and|&', value)

                        for author_name in authors:
                            author_name = author_name.strip()
                            if len(author_name) < 4:
                                continue
                            if author_name in entity:
                                author_id = entity.index(author_name)
                            else:
                                entity.append(author_name)
                                author_id = len(entity) - 1
                            triples.append((entity_id, author_id, relation_id))
                    elif value in entity:
                        value_id = entity.index(value)
                        triples.append((entity_id, value_id, relation_id))
                    else:
                        entity.append(value)
                        value_id = len(entity) - 1;
                        triples.append((entity_id, value_id, relation_id))

        logger.info("Number of entitiesA: %d", len(entityA))
        logger.info("Number of entitiesB: %d", len(entityB))
        logger.info("Number of relationsA: %d", len(relationA))
        logger.info("Number of relationsB: %d", len(relationB))
        logger.info("Number of TriplesA: %d", len(triplesA))
        logger.info("Number of TriplesB: %d", len(triplesB))

        #Extract candidate links and true links
        entity_pairs = []
        true_pairs = []
        for a in record_ids_A:
            a_id = enitity_id_mapping[str(a)]
            for b in record_ids_B:
                b_id = enitity_id_mapping[str(b)]
                entity_pairs.append((a_id,b_id))
                if dni_mapping_A[str(a_id)] == dni_mapping_B[str(b_id)]:
                    true_pairs.append((a_id,b_id))

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

        relation_value_map = {'author': [], 'publisher': [], 'date': [], 'title': [], 'journal': [],
                                'volume': [], 'pages': [], 'address': [] }
        relation = ['author', 'publisher', 'date', 'title', 'journal', 'volume', 'pages', 'address']
        train_triples = []
        val_triples = []
        test_triples = []

        for (true_links, datasetA, datasetB, triples) in [
            (self.true_links, self.trainDataA, self.trainDataB, train_triples),
            (self.true_val_links, self.valDataA, self.valDataB, val_triples),
            (self.true_test_links, self.testDataA, self.testDataB, test_triples)]:

            for (index_a, index_b) in true_links:
                row_a = datasetA.loc[index_a]
                row_b = datasetB.loc[index_b]

                for rel_index in range(len(relation)):
                    rel = relation[rel_index]
                    values = relation_value_map[rel]

                    if row_a[rel].strip() and row_b[rel].strip():
                        if row_a[rel] in values:
                            val_index_a = values.index(row_a[rel])
                        else:
                            values.append(row_a[rel])
                            val_index_a = len(values) - 1

                        if row_b[rel] in values:
                            val_index_b = values.index(row_b[rel])
                        else:
                            values.append(row_b[rel])
                            val_index_b = len(values) - 1

                        if val_index_a != val_index_b:
                            triples.append((val_index_a, val_index_b, rel_index))


        logger.info("Number of Values: %d", str([len(relation_value_map[r]) for r in relation]))
        logger.info("Number of relations: %d", len(relation))
        logger.info("Number of Train Triples: %d", len(train_triples))
        logger.info("Number of Val Triples: %d", len(val_triples))
        logger.info("Number of Test Triples: %d", len(test_triples))

        return (relation_value_map, relation, train_triples, val_triples, test_triples)

    def get_entity_information(self, entity_name):
        try:
            ent_id = entity_name.split('_')[0][4:]
            logger.debug("Searching for entity id: %s", ent_id)
        except Exception as e:
            logger.error(e)
            logger.error("Failed to get entity id for %s", str(entity_name))
            return None

        for dataset in [self.trainDataA, self.trainDataB, self.testDataA, self.testDataB]:
            e = [e for e in dataset.iterrows() if e[1]['id'] == ent_id]
            if len(e):
                return e[0]
        return None

    def get_entity_names(self, dataset):
        return ['cora' + e[1]['id'] for e in dataset.iterrows()]

    def __str__(self):
        return config.CORA_FILE_PREFIX