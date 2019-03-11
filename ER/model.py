import config
import pandas as pd

from common import get_logger

logger = get_logger('RL.ER.GRAPH_ER')

class Graph_ER(object):
    #Knowledge Graph Static file names
    entity_file_suffix = '_entity_id.txt'
    relation_file_suffix = '_relation_id.txt'
    triples_file_suffix = '_triples.txt'
    entity_pairs_files_suffix = '_entity_pairs.txt'
    true_pairs_files_suffix = '_true_pairs.txt'


    def __init__(self, dataset, rebuild=False):
        #Knowledg Graph Data structures
        self.entity = None # List of all unique Entities e.g. Cora1, 1994, etc
        self.relation = None # List of all unique Relationships e.g. author, year, etc.
        self.triples = None # List of triplets of form (entity_id, entity_id, relation_id)
        self.entity_pairs = None # List of tuples of form (entity_id, entity_id). Hypothesis space.
        self.true_pairs = None # List of tuples of form (entity_id, entity_id). Ground Truth.

        self.model = dataset()
        self.dataset_prefix = config.BASE_ER_GRAPH_FOLDER + str(self.model)
        if rebuild:
            self.rebuild_graph()
        else:
            self.load_kg_er_model()

    def rebuild_graph(self):
        entity, relation, triples, entity_pairs, true_pairs = self.model.get_er_model()
        return self.export_kg_er_model(entity, relation, triples, entity_pairs, true_pairs)

    def load_kg_er_model(self):
        try:
            with open(self.dataset_prefix + self.entity_file_suffix, "r") as f:
                self.entity = [e.strip() for e in f.readlines()]
            with open(self.dataset_prefix + self.relation_file_suffix, "r") as f:
                self.relation = [r.strip() for r in f.readlines()]
            with open(self.dataset_prefix + self.triples_file_suffix, "r") as f:
                self.triples = [t.strip().split(',') for t in f.readlines()]
                self.triples = [(int(h), int(t), int(r)) for (h,t,r) in self.triples]
            with open(self.dataset_prefix + self.entity_pairs_files_suffix, "r") as f:
                self.entity_pairs = [ep.strip().split(',') for ep in f.readlines()]
                self.entity_pairs = [(int(h), int(t)) for (h,t) in self.entity_pairs]
            with open(self.dataset_prefix + self.true_pairs_files_suffix, "r") as f:
                self.true_pairs = [tp.strip().split(',') for tp in f.readlines()]
                self.true_pairs = [(int(h), int(t)) for (h,t) in self.true_pairs]
                self.true_pairs = pd.MultiIndex.from_tuples(self.true_pairs)

            logger.info("ER Graph loaded for %s. Entity: %d, Realtion: %d, Triples: %d",
                self.dataset_prefix, len(self.entity), len(self.relation), len(self.triples))
            return (self.entity, self.relation, self.triples, self.entity_pairs, self.true_pairs)
        except IOError:
            logger.error("ER Graph files Not Found. Rebuilding Graph.....")
            return self.rebuild_graph()

    def export_kg_er_model(self, entity, relation, triples, entity_pairs, true_pairs):
        #Generate file names
        e_file = self.dataset_prefix + self.entity_file_suffix
        r_file = self.dataset_prefix + self.relation_file_suffix
        t_file = self.dataset_prefix + self.triples_file_suffix
        ep_file = self.dataset_prefix + self.entity_pairs_files_suffix
        tp_file = self.dataset_prefix + self.true_pairs_files_suffix

        #Write Knowledge Graph to files
        with open(e_file, "w+") as f:
            for e in entity:
                try:
                    f.write("%s\n" % str(e).replace('\n', ' '))
                except:
                    f.write("%s\n" % str(e.encode('ascii', 'ignore').decode('ascii')).replace('\n', ' '))

        with open(r_file, "w+") as f:
            for r in relation:
                try:
                    f.write("%s\n" % str(r).replace('\n', ' '))
                except:
                    f.write("%s\n" % str(r.encode('ascii', 'ignore').decode('ascii')).replace('\n', ' '))

        with open(t_file, "w+") as f:
            for (h,t,r) in triples:
                f.write("%d,%d,%d\n" % (h,t,r))

        with open(ep_file, "w+") as f:
            for (h,t) in entity_pairs:
                f.write("%d,%d\n" % (h,t))

        with open(tp_file, "w+") as f:
            for (h,t) in true_pairs:
                f.write("%d,%d\n" % (h,t))

        logger.info("ER Graph exported for %s", str(self.dataset_prefix))

        #Update current object.
        self.entity = entity
        self.relation = relation
        self.triples = triples
        self.entity_pairs = entity_pairs
        self.true_pairs = true_pairs
        return (self.entity, self.relation, self.triples, self.entity_pairs, self.true_pairs)