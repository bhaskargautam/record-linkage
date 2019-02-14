import config
import pandas as pd

from common import get_logger

logger = get_logger('GRAPH_EAR')

class Graph_EAR(object):
    entity = None
    attribute = None
    relation = None
    value = None
    atriples = None
    rtriples = None
    entity_pairs = None
    true_pairs = None
    dataset_prefix = None
    entity_file_suffix = '_ear_entity_id.txt'
    attribute_file_suffix = '_ear_attribute_id.txt'
    relation_file_suffix = '_ear_relation_id.txt'
    value_file_suffix = '_ear_value_id.txt'
    atriples_file_suffix = '_ear_atriples.txt'
    rtriples_file_suffix = '_ear_rtriples.txt'
    entity_pairs_files_suffix = '_ear_entity_pairs.txt'
    true_pairs_files_suffix = '_ear_true_pairs.txt'

    def __init__(self, dataset_prefix):
        self.dataset_prefix = config.BASE_EAR_GRAPH_FOLDER + dataset_prefix

    def load_kg_ear_model(self):
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
        return True


