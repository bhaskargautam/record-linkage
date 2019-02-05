import tensorflow as tf
import numpy as np

from common import sigmoid, get_logger
from scipy import spatial

logger = get_logger('TransH')

class TransH(object):
    """
        Tensorflow based implementation of TransE method
        User train method to update the embeddings.
    """

    def __init__(self, entity, relation, triples, dimension=10, batchSize=100,
                    learning_rate=0.1, margin=1):
        logger.info("Begin generating TransH embeddings with dimension : %d" ,dimension)

        self.dimension = dimension #Embedding Dimension
        self.batchSize = batchSize #BatchSize for Stochastic Gradient Decent
        self.learning_rate = learning_rate #Learning rate for optmizer
        self.margin = margin #margin or bias used for loss computation
        self.entity = entity #List of entities in Knowledge graph
        self.relation = relation #List of relationships in Knowledge Graph

        # List of triples. Remove last incomplete batch if any.
        self.triples = np.array(triples[0: (len(triples) - len(triples)%batchSize)])
        self.ntriples = np.array(self._get_negative_samples(triples, entity))
        logger.info("Shape of triples: %s", str(self.triples.shape))
        logger.info("Shape of neg triples: %s", str(self.ntriples.shape))

        #Define Embedding Variables
        self.ent_embeddings = tf.get_variable(name = "ent_embeddings", shape = [len(entity), dimension],
                                    initializer = tf.contrib.layers.xavier_initializer(uniform = False))
        self.rel_embeddings = tf.get_variable(name = "rel_embeddings", shape = [len(relation), dimension],
                                    initializer = tf.contrib.layers.xavier_initializer(uniform = False))
        self.norm_vector = tf.get_variable(name = "norm_vector", shape = [len(relation), dimension],
                                    initializer = tf.contrib.layers.xavier_initializer(uniform = False))

        #Define Placeholders for input
        self.head = tf.placeholder(tf.int32, shape=[self.batchSize])
        self.tail = tf.placeholder(tf.int32, shape=[self.batchSize])
        self.rel = tf.placeholder(tf.int32, shape=[self.batchSize])
        self.neg_head = tf.placeholder(tf.int32, shape=[self.batchSize])

        #Load Embedding Vectors
        pos_h = tf.nn.embedding_lookup(self.ent_embeddings, self.head)
        pos_t = tf.nn.embedding_lookup(self.ent_embeddings, self.tail)
        pos_r = tf.nn.embedding_lookup(self.rel_embeddings, self.rel)
        pos_nh = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_head)
        pos_norm = tf.nn.embedding_lookup(self.norm_vector, self.rel)

        #Normalize embedding vectors
        pos_h = tf.nn.l2_normalize(pos_h, 1)
        pos_t = tf.nn.l2_normalize(pos_t, 1)
        pos_r = tf.nn.l2_normalize(pos_r, 1)
        pos_nh = tf.nn.l2_normalize(pos_nh, 1)
        pos_norm = tf.nn.l2_normalize(pos_norm, 1)

        #Project entities to hyperplane
        pos_h = self._transfer(pos_h, pos_norm)
        pos_t = self._transfer(pos_t, pos_norm)
        pos_nh = self._transfer(pos_nh, pos_norm)

        #Compute Loss
        _p_score = self._calc(pos_h, pos_t, pos_r)
        _n_score = self._calc(pos_nh, pos_t, pos_r)

        p_score = tf.reduce_mean(_p_score)
        n_score = tf.reduce_mean(_n_score)
        self.loss = tf.reduce_mean(tf.maximum(p_score - n_score + self.margin, 0))

        #Configure optimizer
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

        #Configure session
        self.sess = tf.Session()

    def _calc(self, h, t, r):
        """
            TransE objective function.
            It estimates embeddings as translation from head to tail entity.
        """
        return abs(h + r - t)

    def _transfer(self, e, n):
        return e - tf.reduce_sum(e * n, 1, keep_dims = True) * n

    def _get_negative_samples(self, triples, entity):
        #Collect negetive samples
        #Todo: use neg_tail and neg_rel also.
        ntriples = []
        all_heads = range(0, len(entity))
        np.random.shuffle(all_heads)
        tuple_triples = set(map(tuple, triples))
        for (h, r, t) in triples:
            for neg_head in all_heads:
                if neg_head == h:
                    continue
                if (neg_head, r, t) not in tuple_triples:
                    ntriples.append((neg_head, r, t))
                    break
        logger.info("Number of negative triples: %d", len(ntriples))
        return ntriples

    def train(self, max_epochs=100):
        """
            Method to train the model and update the embeddings.
        """
        loss = 0
        with self.sess.as_default():
            self.sess.run(tf.global_variables_initializer())

            for epoch in range(0, max_epochs):
                loss = 0
                for i in np.arange(0, len(self.triples), self.batchSize):
                    batchend = min(len(self.triples), i + self.batchSize)
                    feed_dict = {
                        self.head : self.triples[i:batchend][:,0],
                        self.tail : self.triples[i:batchend][:,1],
                        self.rel  : self.triples[i:batchend][:,2],
                        self.neg_head : self.ntriples[i:batchend][:,0]
                        }
                    _ , cur_loss = self.sess.run([self.optimizer, self.loss], feed_dict=feed_dict)
                    loss = loss + cur_loss
                logger.info("Epoch: %d Loss: %f", epoch, loss)
        return loss

    def get_ent_embeddings(self):
        with self.sess.as_default():
            return tf.nn.embedding_lookup(self.ent_embeddings, range(0, len(self.entity))).eval()

    def close_tf_session(self):
        tf.reset_default_graph()
        self.sess.close()
