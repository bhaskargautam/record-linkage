import config
import pandas as pd

from common import get_logger

logger = get_logger('GRAPH_ERER')

class Graph_ERER(object):
    entityA = None
    entityB = None
    relationA = None
    relationB = None
    triplesA = None
    triplesB = None
    entity_pairs = None
    prior_pairs = None
    true_pairs = None
    dataset_prefix = None
    entityA_file_suffix = 'ent_ids_1'
    entityB_file_suffix = 'ent_ids_2'
    relationA_file_suffix = 'rel_ids_1'
    relationB_file_suffix = 'rel_ids_2'
    triplesA_file_suffix = 'triples_1'
    triplesB_file_suffix = 'triples_2'
    entity_pairs_files_suffix = 'ref_ent_ids'
    prior_pairs_files_suffix = 'sup_ent_ids'
    true_pairs_files_suffix = 'true_pairs_ent_ids'


    def __init__(self, dataset_prefix):
        self.dataset_prefix = config.BASE_ERER_GRAPH_FOLDER + dataset_prefix + "/"

    def export_kg_erer_model(self, entityA, entityB, relationA, relationB, triplesA,
                                triplesB, entity_pairs, prior_pairs, true_pairs):

        logger.info("Saving ERER Graph for %s", self.dataset_prefix)
        self.entityA = entityA
        self.entityB = entityB
        self.relationA = relationA
        self.relationB = relationB
        self.triplesA = triplesA
        self.triplesB = triplesB
        self.entity_pairs = entity_pairs
        self.prior_pairs = prior_pairs
        self.true_pairs = true_pairs

        #Write Files
        with open(self.dataset_prefix + self.entityA_file_suffix, "w") as f:
            for index in range(0, len(entityA)):
                try:
                    f.write("%d\t%s\n" % (index, str(entityA[index]).replace('\n', ' ').strip()))
                except UnicodeEncodeError:
                    f.write("%d\t%s\n" % (index, str(entityA[index].encode('ascii', 'ignore').decode('ascii')).replace('\n', ' ').strip()))

        with open(self.dataset_prefix + self.entityB_file_suffix, "w") as f:
            for index in range(0, len(entityB)):
                try:
                    f.write("%d\t%s\n" % (index, str(entityB[index]).replace('\n', ' ').strip()))
                except UnicodeEncodeError:
                    f.write("%d\t%s\n" % (index, str(entityB[index].encode('ascii', 'ignore').decode('ascii')).replace('\n', ' ').strip()))

        with open(self.dataset_prefix + self.entity_pairs_files_suffix, "w") as f:
            for e1, e2 in entity_pairs:
                f.write("%d\t%d\n" % (e1, e2))

        with open(self.dataset_prefix + self.relationA_file_suffix, "w") as f:
            for index in range(0, len(relationA)):
                f.write("%d\t%s\n" % (index, str(relationA[index]).replace('\n', ' ').strip()))

        with open(self.dataset_prefix + self.relationB_file_suffix, "w") as f:
            for index in range(0, len(relationB)):
                f.write("%d\t%s\n" % (index, str(relationB[index]).replace('\n', ' ').strip()))

        with open(self.dataset_prefix + self.triplesA_file_suffix, "w") as f:
            for index in range(0, len(triplesA)):
                f.write("%d\t%d\t%d\n" % (triplesA[index][0], triplesA[index][2], triplesA[index][1]))

        with open(self.dataset_prefix + self.triplesB_file_suffix, "w") as f:
            for index in range(0, len(triplesB)):
                f.write("%d\t%d\t%d\n" % (triplesB[index][0], triplesB[index][2], triplesB[index][1]))

        with open(self.dataset_prefix + self.prior_pairs_files_suffix, "w") as f:
            for e1, e2 in prior_pairs:
                f.write("%d\t%d\n" % (e1, e2))

        with open(self.dataset_prefix + self.true_pairs_files_suffix, "w") as f:
            for e1, e2 in true_pairs:
                f.write("%d\t%d\n" % (e1, e2))

        logger.info("Successfully ERER graph for %s", str(self.dataset_prefix))
        return True

    def load_kg_erer_model(self):
        #Initialze graph
        self.entityA = []
        self.entityB = []
        self.relationA = []
        self.relationB = []
        self.triplesA = []
        self.triplesB = []
        self.entity_pairs = []
        self.true_pairs = []
        self.prior_pairs = []

        #Read Files
        with open(self.dataset_prefix + self.entityA_file_suffix, "r") as f:
            for line in f.readlines():
                self.entityA.append(line.split('\t')[1].strip())

        with open(self.dataset_prefix + self.entityB_file_suffix, "r") as f:
            for line in f.readlines():
                self.entityB.append(line.split('\t')[1].strip())

        with open(self.dataset_prefix + self.entity_pairs_files_suffix, "r") as f:
            for line in f.readlines():
                self.entity_pairs.append((line.split('\t')[0].strip(), line.split('\t')[1].strip()))

        with open(self.dataset_prefix + self.relationA_file_suffix, "r") as f:
            for line in f.readlines():
                self.relationA.append(line.split('\t')[1].strip())

        with open(self.dataset_prefix + self.relationB_file_suffix, "r") as f:
            for line in f.readlines():
                self.relationB.append(line.split('\t')[1].strip())

        with open(self.dataset_prefix + self.triplesA_file_suffix, "r") as f:
            for line in f.readlines():
                splits = line.split('\t')
                self.triplesA.append((splits[0].strip(), splits[2].strip(), splits[1].strip()))

        with open(self.dataset_prefix + self.triplesB_file_suffix, "r") as f:
            for line in f.readlines():
                splits = line.split('\t')
                self.triplesB.append((splits[0].strip(), splits[2].strip(), splits[1].strip()))

        with open(self.dataset_prefix + self.prior_pairs_files_suffix, "r") as f:
            for line in f.readlines():
                self.prior_pairs.append((line.split('\t')[0].strip(), line.split('\t')[1].strip()))

        with open(self.dataset_prefix + self.true_pairs_files_suffix, "r") as f:
            for line in f.readlines():
                self.true_pairs.append((line.split('\t')[0].strip(), line.split('\t')[1].strip()))

        self.true_pairs = pd.MultiIndex.from_tuples(self.true_pairs)
        logger.info("Successfully Loaded ERER graph for %s", self.dataset_prefix)
        return (self.entityA, self.entityB, self.relationA, self.relationB,
                    self.triplesA, self.triplesB, self.entity_pairs, self.prior_pairs, self.true_pairs)

    def get_er_model(self):
        if not self.entityA:
            #Load Graph if not avaible before
            self.load_kg_erer_model()

        entity = list(set(self.entityA + self.entityB))
        relation = list(set(self.relationA + self.relationB))
        logger.info("All relations: %s", str(relation))

        #Create index for enitities
        ent_indexA = {}
        ent_indexB = {}
        for e in self.entityA:
            ent_indexA[e] = entity.index(e)
        for e in self.entityB:
            ent_indexB[e] = entity.index(e)

        #Extract Triples
        triples = []
        for (h, t, r) in self.triplesA:
            triples.append((ent_indexA[self.entityA[int(h)]],
                            ent_indexA[self.entityA[int(t)]],
                            relation.index(self.relationA[int(r)])))

        for (h, t, r) in self.triplesB:
            triples.append((ent_indexB[self.entityB[int(h)]],
                            ent_indexB[self.entityB[int(t)]],
                            relation.index(self.relationB[int(r)])))

        #Add triples of form (e1, e2, 'same_as')
        same_as_relation_id = relation.index("same_as")
        logger.info("No. of triples before prior alignment: %d", len(triples))
        for (h, t) in self.prior_pairs:
            triples.append((ent_indexA[self.entityA[int(h)]],
                                ent_indexB[self.entityB[int(t)]],
                                same_as_relation_id))
        logger.info("After Adding %d prior alignment pairs: %d", len(self.prior_pairs), len(triples))

        #Extract Entity pairs
        ep = []
        for (ea, eb) in self.entity_pairs:
            ep.append((ent_indexA[self.entityA[int(ea)]],
                        ent_indexB[self.entityB[int(eb)]]))

        new_tp = []
        for (ea, eb) in self.true_pairs:
            new_tp.append((ent_indexA[self.entityA[int(ea)]],
                        ent_indexB[self.entityB[int(eb)]]))

        logger.info("Number of entities: %d", len(entity))
        logger.info("Number of relations: %d", len(relation))
        logger.info("Number of Triples: %d", len(triples))

        true_pairs = pd.MultiIndex.from_tuples(new_tp)
        logger.info("Number of entity pairs: %d", len(self.entity_pairs))
        logger.info("Number of true pairs: %d", len(true_pairs))

        return (entity, relation, triples, ep, true_pairs)
