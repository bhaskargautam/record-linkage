import config
import pandas as pd

from common import get_logger

logger = get_logger('RL.ER.GRAPH_ER')

class Graph_ER(object):
    entity = None
    relation = None
    triples = None
    entity_pairs = None
    true_pairs = None
    dataset_prefix = None
    entity_file_suffix = '_entity_id.txt'
    relation_file_suffix = '_relation_id.txt'
    triples_file_suffix = '_triples.txt'
    entity_pairs_files_suffix = '_entity_pairs.txt'
    true_pairs_files_suffix = '_true_pairs.txt'


    def __init__(self, dataset_prefix):
        self.dataset_prefix = config.BASE_ER_GRAPH_FOLDER + dataset_prefix

    def load_kg_er_model(self):
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


    def export_kg_er_model(self, entity, relation, triples, entity_pairs, true_pairs):
        e_file = self.dataset_prefix + self.entity_file_suffix
        r_file = self.dataset_prefix + self.relation_file_suffix
        t_file = self.dataset_prefix + self.triples_file_suffix
        ep_file = self.dataset_prefix + self.entity_pairs_files_suffix
        tp_file = self.dataset_prefix + self.true_pairs_files_suffix

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
        return True