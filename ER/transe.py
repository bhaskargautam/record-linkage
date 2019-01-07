import numpy as np
import math

from common import sigmoid, get_logger
from scipy import spatial

logger = get_logger('TransE')

class TransE(object):
    """class to generate TransE embeddings"""
    dimension = None
    learning_rate = None
    margin = None
    norm = None
    ent_embedding = None
    rel_embedding = None
    ent_embedding_tmp = None
    rel_embedding_tmp = None

    def __init__(self, entity, relation, triples, dimension=80, batchSize=100,
                    learning_rate=0.1, margin=1, max_epochs=10, norm='L1'):
        logger.info("Begin generating TransE embeddings with dimension : %d" ,dimension)
        self.dimension = dimension
        self.learning_rate = learning_rate
        self.margin = margin
        self.norm = norm
        low_bound = -6/math.sqrt(self.dimension)
        high_bound = 6/math.sqrt(self.dimension)
        self.ent_embedding = np.random.uniform(low=low_bound, high=high_bound, size=(len(entity), dimension))
        self.rel_embedding = np.random.uniform(low=low_bound, high=high_bound, size=(len(relation), dimension))
        self.ent_embedding_tmp = np.random.rand(len(entity), dimension)
        self.rel_embedding_tmp = np.random.rand(len(relation), dimension)

        self.ent_embedding = self.normalize(self.ent_embedding)
        self.rel_embedding = self.normalize(self.rel_embedding)

        #Collect negetive samples
        ntriples = []
        for (h, r, t) in triples:
            neg_count = 0
            all_neg_heads = range(0, h) + range(h + 1, len(entity))
            np.random.shuffle(all_neg_heads)
            for neg_head in all_neg_heads:
                if (neg_head, r, t) not in triples:
                    ntriples.append((neg_head, r, t))
                    break
        logger.info("Number of negative triples: %d", len(ntriples))

        #BEGIN SGD OPTIMIZATION
        for epoch in range(0, max_epochs):
            loss = 0

            for i in np.arange(0, len(triples), batchSize):
                for j in np.arange(0, min(batchSize, len(triples) - i)):
                    pos_dist = self.distance(triples[i+j][0], triples[i+j][1], triples[i+j][2])
                    neg_dist = self.distance(ntriples[i+j][0], ntriples[i+j][1], ntriples[i+j][2])
                    loss = loss + pos_dist + (1 - neg_dist) + self.margin

                    tmp_pos = []
                    tmp_neg = []
                    delta_pos = 2 * self.learning_rate * (
                                self.ent_embedding[triples[i+j][1]] -
                                self.ent_embedding[triples[i+j][0]] -
                                self.rel_embedding[triples[i+j][2]])
                    delta_neg = 2 * self.learning_rate * (
                                self.ent_embedding[ntriples[i+j][1]] -
                                self.ent_embedding[ntriples[i+j][0]] -
                                self.rel_embedding[ntriples[i+j][2]])
                    if self.norm == 'L1':
                        for i in range(0, self.dimension):
                            if delta_pos[i] > 0:
                                tmp_pos.append(1)
                            else:
                                tmp_pos.append(-1)
                            if delta_neg[i] > 0:
                                tmp_neg.append(1)
                            else:
                                tmp_neg.append(-1)
                    elif self.norm == 'L2':
                        tmp_pos = delta_pos
                        tmp_neg = delta_neg
                    else:
                        raise Exception("Norm is configured incorrectly. Should be either L1 or L2.")

                    self.ent_embedding_tmp[triples[i+j][0]] = self.ent_embedding[triples[i+j][0]] + (tmp_pos*pos_dist)
                    self.ent_embedding_tmp[triples[i+j][1]] = self.ent_embedding[triples[i+j][1]] + (tmp_pos*pos_dist) - (tmp_neg*(1-neg_dist))
                    self.rel_embedding_tmp[triples[i+j][2]] = self.rel_embedding[triples[i+j][2]] + (tmp_pos*pos_dist) - (tmp_neg*(1-neg_dist))
                    self.ent_embedding_tmp[ntriples[i+j][0]] = self.ent_embedding[triples[i+j][0]] - (tmp_neg*(1-neg_dist))
                    #self.ent_embedding_tmp[ntriples[i+j][1]] = self.ent_embedding[triples[i+j][1]] + tmp_neg
                    """
                    Debug internal steps
                    logger.info("\n\n Head: %s", str(self.ent_embedding[triples[i+j][0]]))
                    logger.info("Tail: %s", str(self.ent_embedding[triples[i+j][1]]))
                    logger.info("Rel: %s", str(self.rel_embedding[triples[i+j][2]]))
                    logger.info("Neg Head: %s", str(self.ent_embedding[ntriples[i+j][0]]))
                    logger.info("Updated Head: %s", str(self.ent_embedding_tmp[triples[i+j][0]]))
                    logger.info("Updated Tail: %s", str(self.ent_embedding_tmp[triples[i+j][1]]))
                    logger.info("Updated Rel: %s", str(self.rel_embedding_tmp[triples[i+j][2]]))
                    logger.info("Updated Neg Head: %s", str(self.ent_embedding_tmp[ntriples[i+j][0]]))

                    logger.info("True Prev. dist: %f New dist %f", pos_dist, self.distance_tmp(triples[i+j][0], triples[i+j][1], triples[i+j][2]))
                    logger.info("False Prev. dist: %f New dist %f", neg_dist, self.distance_tmp(ntriples[i+j][0], ntriples[i+j][1], ntriples[i+j][2]))
                    """
            self.ent_embedding = self.normalize(self.ent_embedding_tmp)
            self.rel_embedding = self.normalize(self.rel_embedding_tmp)

            logger.info("Epoch: %d Loss: %f", epoch, loss)

    def normalize(self, arr):
        x = np.linalg.norm(arr)
        if x == 0:
            return arr
        return arr / x

    def distance(self, head, tail, rel, norm='L1'):
        if self.norm == 'L1':
            dist = self.ent_embedding[head] + self.rel_embedding[rel] - self.ent_embedding[tail]
            return abs(dist).sum()
        elif self.norm == 'L2':
            dist = self.ent_embedding[head] + self.rel_embedding[rel] - self.ent_embedding[tail]
            return (dist*dist).sum()
        else:
            raise Exception("Norm is configured incorrectly. Should be either L1 or L2.")

    def distance_tmp(self, head, tail, rel, norm='L1'):
        if self.norm == 'L1':
            dist = self.ent_embedding_tmp[head] + self.rel_embedding_tmp[rel] - self.ent_embedding_tmp[tail]
            return abs(dist).sum()
        elif self.norm == 'L2':
            dist = self.ent_embedding_tmp[head] + self.rel_embedding_tmp[rel] - self.ent_embedding_tmp[tail]
            return (dist*dist).sum()
        else:
            raise Exception("Norm is configured incorrectly. Should be either L1 or L2.")

    def similarity(self, head, tail, rel=None):
        return sigmoid(abs(spatial.distance.cosine(
            self.ent_embedding[head],
            self.ent_embedding[tail])))

    def similarity_tmp(self, head, tail, rel):
        return sigmoid(abs(spatial.distance.cosine(
            self.ent_embedding_tmp[head] + self.rel_embedding_tmp[rel],
            self.ent_embedding_tmp[tail])))

    def update_rel(self, head, tail, rel, y):
        sim = self.similarity(head, tail, rel)
        if y < sim:
            for dim in range(0, self.dimension):
                self.ent_embedding_tmp[head][dim] = self.ent_embedding[head][dim] + self.learning_rate*(sim - y)
                self.ent_embedding_tmp[tail][dim] = self.ent_embedding[tail][dim] + self.learning_rate*(sim - y)
                self.rel_embedding_tmp[rel][dim] = self.rel_embedding[rel][dim] + self.learning_rate*(sim - y)
        else:
            for dim in range(0, self.dimension):
                self.ent_embedding_tmp[head][dim] = self.ent_embedding[head][dim] - self.learning_rate*(sim - y)
                self.ent_embedding_tmp[tail][dim] = self.ent_embedding[tail][dim] - self.learning_rate*(sim - y)
                self.rel_embedding_tmp[rel][dim] = self.rel_embedding[rel][dim] - self.learning_rate*(sim - y)
        logger.info("Prev sim: %f New Sim %f Expected sim: %d", sim, self.similarity_tmp(head, tail, rel), y)
        return abs(y - sim)