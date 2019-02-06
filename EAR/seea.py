import numpy as np
import tensorflow as tf

from common import sigmoid, get_logger
from scipy import spatial

logger = get_logger('SEEA')

class SEEA(object):

    def __init__(self, entity, attribute, relation, value, atriples, rtriples,
            dimension=10, learning_rate=0.1, batchSize=100, margin=1):
        self.entity = entity
        self.attribute = attribute
        self.relation = relation
        self.value = value
        self.dimension = dimension
        self.learning_rate = learning_rate
        self.batchSize = batchSize
        self.margin = margin

        # List of triples. Remove last incomplete batch if any.
        self.atriples = np.array(atriples[0: (len(atriples) - len(atriples)%batchSize)])
        self.rtriples = np.array(rtriples[0: (len(rtriples) - len(rtriples)%batchSize)])
        logger.info("Modified Atriples size: %d", len(self.atriples))
        logger.info("Modified Rtriples size: %d", len(self.rtriples))

        #Collect Negative Samples
        self.natriples = np.array(self._get_negative_samples(self.atriples, self.entity))
        self.nrtriples = np.array(self._get_negative_samples(self.rtriples, self.entity))

        #Define Embedding Variables
        self.ent_embeddings = tf.get_variable(name = "ent_embeddings", shape = [len(entity), dimension],
                                    initializer = tf.contrib.layers.xavier_initializer(uniform = False))
        self.rel_embeddings = tf.get_variable(name = "rel_embeddings", shape = [len(relation), dimension],
                                    initializer = tf.contrib.layers.xavier_initializer(uniform = False))
        self.attr_embeddings = tf.get_variable(name = "attr_embeddings", shape = [len(attribute), dimension],
                                    initializer = tf.contrib.layers.xavier_initializer(uniform = False))
        self.val_embeddings = tf.get_variable(name = "val_embeddings", shape = [len(value), dimension],
                                    initializer = tf.contrib.layers.xavier_initializer(uniform = False))
        self.projection_matrix = tf.get_variable(name = "projection_matrix", shape = [len(attribute), dimension],
                                    initializer = tf.contrib.layers.xavier_initializer(uniform = False))

        #Define Placeholders for input
        self.head = tf.placeholder(tf.int32, shape=[self.batchSize])
        self.tail = tf.placeholder(tf.int32, shape=[self.batchSize])
        self.rel = tf.placeholder(tf.int32, shape=[self.batchSize])
        self.neg_head = tf.placeholder(tf.int32, shape=[self.batchSize])

        self.attr_head = tf.placeholder(tf.int32, shape=[self.batchSize])
        self.val = tf.placeholder(tf.int32, shape=[self.batchSize])
        self.attr = tf.placeholder(tf.int32, shape=[self.batchSize])
        self.neg_attr_head = tf.placeholder(tf.int32, shape=[self.batchSize])

        #Load Embedding Vectors
        pos_h = tf.nn.embedding_lookup(self.ent_embeddings, self.head)
        pos_t = tf.nn.embedding_lookup(self.ent_embeddings, self.tail)
        pos_r = tf.nn.embedding_lookup(self.rel_embeddings, self.rel)
        pos_nh = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_head)
        pos_attr_h = tf.nn.embedding_lookup(self.ent_embeddings, self.attr_head)
        pos_val = tf.nn.embedding_lookup(self.val_embeddings, self.val)
        pos_attr = tf.nn.embedding_lookup(self.attr_embeddings, self.attr)
        pos_attr_nh = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_attr_head)
        pos_proj = tf.nn.embedding_lookup(self.attr_embeddings, self.attr)

        proj_pos_attr_h = self._transfer(pos_attr_h, pos_proj)
        proj_pos_attr_nh = self._transfer(pos_attr_nh, pos_proj)

        #Compute Loss
        _p_score = self._calc(pos_h, pos_t, pos_r)
        _n_score = self._calc(pos_nh, pos_t, pos_r)

        _ap_score = self._attr_calc(proj_pos_attr_h, pos_val, pos_attr)
        _an_score = self._attr_calc(proj_pos_attr_nh, pos_val, pos_attr)

        p_score = tf.reduce_mean(_p_score)
        n_score = tf.reduce_mean(_n_score)
        ap_score = tf.reduce_mean(_ap_score)
        an_score = tf.reduce_mean(_an_score)
        self.rel_loss = tf.reduce_mean(tf.maximum(p_score - n_score + self.margin, 0))
        self.attr_loss = tf.reduce_mean(tf.maximum(ap_score - an_score + self.margin, 0))

        #Configure optimizer
        self.rel_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.rel_loss)
        self.attr_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.attr_loss)

         #Configure session
        self.sess = tf.Session()

    def _calc(self, h, t, r):
        """
            TransE objective function.
            It estimates embeddings as translation from head to tail entity.
        """
        return abs(h + r - t)

    def _attr_calc(self, h, v, a):
        return abs(tf.nn.tanh(h + a) - v)

    def _transfer(self, e, n):
        return e - tf.reduce_sum(e * n, 1, keep_dims = True) * n

    def train(self, max_epochs=100):
        loss = 0
        with self.sess.as_default():
            self.sess.run(tf.global_variables_initializer())
            for epoch in range(0, max_epochs):
                rel_loss = attr_loss = 0

                #Relation Triple Encoder
                for i in np.arange(0, len(self.rtriples), self.batchSize):
                    batchend = min(len(self.rtriples), i + self.batchSize)
                    feed_dict = {
                        self.head : self.rtriples[i:batchend][:,0],
                        self.tail : self.rtriples[i:batchend][:,1],
                        self.rel : self.rtriples[i:batchend][:,2],
                        self.neg_head : self.nrtriples[i:batchend][:,0]
                    }
                    _, cur_rel_loss = self.sess.run([self.rel_optimizer, self.rel_loss],
                        feed_dict=feed_dict)
                    #logger.info("Cur rel loss: %f", cur_rel_loss)
                    rel_loss = rel_loss + cur_rel_loss

                #Attributional Triple Encoder
                for i in np.arange(0, len(self.atriples), self.batchSize):
                    batchend = min(len(self.atriples), i + self.batchSize)
                    feed_dict = {
                        self.attr_head : self.atriples[i:batchend][:,0],
                        self.val : self.atriples[i:batchend][:,1],
                        self.attr : self.atriples[i:batchend][:,2],
                        self.neg_attr_head : self.natriples[i:batchend][:,0]
                    }
                    _, cur_attr_loss = self.sess.run([self.attr_optimizer, self.attr_loss],
                        feed_dict=feed_dict)
                    #logger.info("Cur attr loss: %f", cur_attr_loss)
                    attr_loss = attr_loss + cur_attr_loss

                loss = rel_loss + attr_loss
                logger.info("Epoch: %d Loss: %f Rel_loss: %f, Attr_loss: %f",
                    epoch, loss, rel_loss, attr_loss)

        return loss

    def _get_negative_samples(self, triples, entity):
        #Collect negetive samples
        #Todo: use neg_tail and neg_rel also.
        ntriples = []
        all_heads = range(0, len(entity))
        np.random.shuffle(all_heads)
        tuple_triples = set(map(tuple, triples))
        logger.info("SIze of tuple_triples: %d", len(tuple_triples))
        for (h, r, t) in triples:
            for neg_head in all_heads:
                if neg_head == h:
                    continue
                if (neg_head, r, t) not in tuple_triples:
                    ntriples.append((neg_head, r, t))
                    break
        logger.info("Number of negative triples: %d", len(ntriples))
        return ntriples

    def get_ent_embeddings(self):
        with self.sess.as_default():
            return tf.nn.embedding_lookup(self.ent_embeddings, range(0, len(self.entity))).eval()

    def close_tf_session(self):
        tf.reset_default_graph()
        self.sess.close()

    def add_rel_triples(self, triples):
        self.rtriples = self.rtriples.tolist() + triples
        np.random.shuffle(self.rtriples)
        self.rtriples = np.array(
            self.rtriples[0: (len(self.rtriples) - len(self.rtriples)%self.batchSize)])

        self.nrtriples = np.array(self._get_negative_samples(self.rtriples, self.entity))
        return len(self.rtriples)

