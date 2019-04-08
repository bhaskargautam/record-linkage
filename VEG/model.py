import config
import pandas as pd

from common import get_logger, create_folder_if_missing

logger = get_logger('RL.ER.GRAPH_VEG.')

class Graph_VEG(object):
    #Knowledge Graph Static file names
    relation_value_map_file_suffix = '_rel_value_map.txt'
    relation_file_suffix = '_relation_id.txt'
    train_triples_file_suffix = '_train_triples.txt'
    val_triples_file_suffix = '_val_triples.txt'
    test_triples_file_suffix = '_test_triples.txt'

    def __init__(self, dataset, rebuild=False):
        #Knowledg Graph Data structures
        self.relation_value_map = {} # Maps relation to List of all unique Values e.g. solter, 1994, etc
        self.relation = None # List of all unique Relationships e.g. author, year, etc.
        self.train_triples = None # List of triplets of form (value_id, value_id, relation_id)
        self.val_triples = None # List of triplets of form (value_id, value_id, relation_id)
        self.test_triples = None # List of triplets of form (value_id, value_id, relation_id)

        self.model = dataset()
        self.dataset_prefix = config.BASE_VEG_GRAPH_FOLDER + str(self.model)
        create_folder_if_missing(self.dataset_prefix)
        if rebuild:
            self.rebuild_graph()
        else:
            self.load_kg_veg_model()

    def rebuild_graph(self):
        relation_value_map, relation, train_triples, val_triples, test_triples = self.model.get_veg_model()
        return self.export_kg_veg_model(relation_value_map, relation, train_triples, val_triples, test_triples)

    def load_kg_veg_model(self):
        try:
            with open(self.dataset_prefix + self.relation_file_suffix, "r") as f:
                self.relation = [r.strip() for r in f.readlines()]
            for r in self.relation:
                self.relation_value_map[r] = []
            with open(self.dataset_prefix + self.relation_value_map_file_suffix, "r") as f:
                for line in [e.strip() for e in f.readlines()]:
                    splits = line.split('\t')
                    if len(splits) < 2:
                        logger.error("Incorrect line: %s", line)
                        continue
                    value = splits[0]
                    rel = splits[1]
                    self.relation_value_map[rel].append(value)
            with open(self.dataset_prefix + self.train_triples_file_suffix, "r") as f:
                self.train_triples = [t.strip().split(',') for t in f.readlines()]
                self.train_triples = [(int(h), int(t), int(r)) for (h,t,r) in self.train_triples]
            with open(self.dataset_prefix + self.val_triples_file_suffix, "r") as f:
                self.val_triples = [t.strip().split(',') for t in f.readlines()]
                self.val_triples = [(int(h), int(t), int(r)) for (h,t,r) in self.val_triples]
            with open(self.dataset_prefix + self.test_triples_file_suffix, "r") as f:
                self.test_triples = [t.strip().split(',') for t in f.readlines()]
                self.test_triples = [(int(h), int(t), int(r)) for (h,t,r) in self.test_triples]

            logger.info("VEG Graph loaded for %s. relation_value_map: %d, Realtion: %d, Train Triples: %d",
                self.dataset_prefix, len(self.relation_value_map), len(self.relation), len(self.train_triples))
            return (self.relation_value_map, self.relation, self.train_triples, self.val_triples, self.test_triples)
        except IOError:
            logger.error("VEG Graph files Not Found. Rebuilding Graph.....")
            return self.rebuild_graph()

    def export_kg_veg_model(self, relation_value_map, relation, train_triples, val_triples, test_triples):
        #Generate file names
        v_file = self.dataset_prefix + self.relation_value_map_file_suffix
        r_file = self.dataset_prefix + self.relation_file_suffix
        tr_file = self.dataset_prefix + self.train_triples_file_suffix
        va_file = self.dataset_prefix + self.val_triples_file_suffix
        te_file = self.dataset_prefix + self.test_triples_file_suffix

        #Write Knowledge Graph to files
        with open(v_file, "w+") as f:
            for r in relation_value_map:
                for v in relation_value_map[r]:
                    try:
                        f.write("%s\t%s\n" % (str(v).replace('\n', ' '), str(r)))
                    except:
                        f.write("%s\t%s\n" % (str(v.encode('ascii', 'ignore').decode('ascii')).replace('\n', ' '), str(r)))

        with open(r_file, "w+") as f:
            for r in relation:
                try:
                    f.write("%s\n" % str(r).replace('\n', ' '))
                except:
                    f.write("%s\n" % str(r.encode('ascii', 'ignore').decode('ascii')).replace('\n', ' '))

        with open(tr_file, "w+") as f:
            for (h,t,r) in train_triples:
                f.write("%d,%d,%d\n" % (h,t,r))

        with open(va_file, "w+") as f:
            for (h,t,r) in val_triples:
                f.write("%d,%d,%d\n" % (h,t,r))

        with open(te_file, "w+") as f:
            for (h,t,r) in test_triples:
                f.write("%d,%d,%d\n" % (h,t,r))

        logger.info("VEG Graph exported for %s", str(self.dataset_prefix))

        #Update current object.
        self.relation_value_map = relation_value_map
        self.relation = relation
        self.train_triples = train_triples
        self.val_triples = val_triples
        self.test_triples = test_triples
        return (self.relation_value_map, self.relation, self.train_triples, self.val_triples, self.test_triples)