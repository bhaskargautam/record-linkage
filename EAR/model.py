import config
import pandas as pd

from common import get_logger

logger = get_logger('RL.EAR.GRAPH_EAR')

class Graph_EAR(object):
    #Knowledg Graph Data structures
    entity = None # List of all unique Entities e.g. Cora1, rec-1-org etc
    attribute = None # List of all unique Attributes e.g. year, author, etc
    relation = None # List of all unique Relation e.g. same_as
    value = None # List of all unique Values/literals e.g. 893, 1994, lehman etc
    atriples = None # List of Attributional Triplets of form (Cora1, 1994, year)
    rtriples = None # List of Relational Triplets of form (Cora1, Cora2, same_as)
    entity_pairs = None # List of tuples of form (entity_id, entity_id). Hypothesis space.
    true_pairs = None # List of tuples of form (entity_id, entity_id). Ground Truth.

    #Knowledge Graph file names
    dataset_prefix = None
    entity_file_suffix = '_ear_entity_id.txt'
    attribute_file_suffix = '_ear_attribute_id.txt'
    relation_file_suffix = '_ear_relation_id.txt'
    value_file_suffix = '_ear_value_id.txt'
    atriples_file_suffix = '_ear_atriples.txt'
    rtriples_file_suffix = '_ear_rtriples.txt'
    entity_pairs_files_suffix = '_ear_entity_pairs.txt'
    true_pairs_files_suffix = '_ear_true_pairs.txt'

    def __init__(self, dataset, rebuild=False):
        self.model = dataset()
        self.dataset_prefix = config.BASE_EAR_GRAPH_FOLDER + str(self.model)
        if rebuild:
            self.rebuild_graph()
        else:
            self.load_kg_ear_model()

    def rebuild_graph(self):
        entity, attribute, relation, value, atriples, rtriples, entity_pairs, \
                    true_pairs = self.model.get_ear_model()
        return self.export_kg_ear_model(entity, attribute, relation, value, \
                    atriples, rtriples, entity_pairs, true_pairs)

    def load_kg_ear_model(self):
        try:
            with open(self.dataset_prefix + self.entity_file_suffix, "r") as f:
                self.entity = [e.strip() for e in f.readlines()]

            with open(self.dataset_prefix + self.attribute_file_suffix, "r") as f:
                self.attribute = [a.strip() for a in f.readlines()]

            with open(self.dataset_prefix + self.relation_file_suffix, "r") as f:
                self.relation = [r.strip() for r in f.readlines()]

            with open(self.dataset_prefix + self.value_file_suffix, "r") as f:
                self.value = [v.strip() for v in f.readlines()]

            with open(self.dataset_prefix + self.atriples_file_suffix, "r") as f:
                self.atriples = [t.strip().split(',') for t in f.readlines()]
                self.atriples = [(int(h), int(a), int(v)) for (h,a,v) in self.atriples]

            with open(self.dataset_prefix + self.rtriples_file_suffix, "r") as f:
                self.rtriples = [t.strip().split(',') for t in f.readlines()]
                self.rtriples = [(int(h), int(t), int(r)) for (h,t,r) in self.rtriples]

            with open(self.dataset_prefix + self.entity_pairs_files_suffix, "r") as f:
                self.entity_pairs = [t.strip().split(',') for t in f.readlines()]
                self.entity_pairs = [(int(h), int(t)) for (h,t) in self.entity_pairs]

            with open(self.dataset_prefix + self.true_pairs_files_suffix, "r") as f:
                self.true_pairs = [t.strip().split(',') for t in f.readlines()]
                self.true_pairs = [(int(h), int(t)) for (h,t) in self.true_pairs]
                self.true_pairs = pd.MultiIndex.from_tuples(self.true_pairs)


            logger.info("Loaded EAR model for %s", self.dataset_prefix)
            logger.info("Entity: %d, Relation: %d, Attribute: %d, Value %d ATriples: %d",
                        len(self.entity), len(self.relation), len(self.attribute),
                        len(self.value), len(self.atriples))
            return (self.entity, self.attribute, self.relation, self.value,
                    self.atriples, self.rtriples, self.entity_pairs, self.true_pairs)
        except IOError:
            logger.error("EAR Graph files Not Found. Rebuilding Graph.....")
            return self.rebuild_graph()

    def export_kg_ear_model(self, entity, attribute, relation, value, atriples,
                                rtriples, entity_pairs, true_pairs):
        with open(self.dataset_prefix + self.entity_file_suffix, "w+") as f:
            for e in entity:
                try:
                    f.write("%s\n" % str(e).replace('\n', ' '))
                except:
                    f.write("%s\n" % str(e.encode('ascii', 'ignore').decode('ascii')).replace('\n', ' '))

        with open(self.dataset_prefix + self.attribute_file_suffix, "w+") as f:
            for a in attribute:
                try:
                    f.write("%s\n" % str(a).replace('\n', ' '))
                except:
                    f.write("%s\n" % str(a.encode('ascii', 'ignore').decode('ascii')).replace('\n', ' '))

        with open(self.dataset_prefix + self.relation_file_suffix, "w+") as f:
            for r in relation:
                try:
                    f.write("%s\n" % str(r).replace('\n', ' '))
                except:
                    f.write("%s\n" % str(r.encode('ascii', 'ignore').decode('ascii')).replace('\n', ' '))

        with open(self.dataset_prefix + self.value_file_suffix, "w+") as f:
            for v in value:
                try:
                    f.write("%s\n" % str(v).replace('\n', ' '))
                except:
                    f.write("%s\n" % str(v.encode('ascii', 'ignore').decode('ascii')).replace('\n', ' '))

        with open(self.dataset_prefix + self.atriples_file_suffix, "w+") as f:
            for (h,a,v) in atriples:
                f.write("%d,%d,%d\n" % (h,a,v))

        with open(self.dataset_prefix + self.rtriples_file_suffix, "w+") as f:
            for (h,t,r) in rtriples:
                f.write("%d,%d,%d\n" % (h,t,r))

        with open(self.dataset_prefix + self.entity_pairs_files_suffix, "w+") as f:
            for (h,t) in entity_pairs:
                f.write("%d,%d\n" % (h,t))

        with open(self.dataset_prefix + self.true_pairs_files_suffix, "w+") as f:
            for (h,t) in true_pairs:
                f.write("%d,%d\n" % (h,t))

        logger.info("Exported EAR Graph for %s", self.dataset_prefix)
        #update current object
        self.entity = entity
        self.attribute = attribute
        self.relation = relation
        self.value = value
        self.atriples = atriples
        self.rtriples = rtriples
        self.entity_pairs = entity_pairs
        self.true_pairs = true_pairs

        return (self.entity, self.attribute, self.relation, self.value,
                    self.atriples, self.rtriples, self.entity_pairs, self.true_pairs)


