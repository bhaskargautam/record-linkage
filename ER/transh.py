import config
import tensorflow as tf
import numpy as np

from common import get_logger, get_negative_samples, get_tf_summary_file_path
from scipy import spatial

logger = get_logger('RL.ER.TransH')

class TransH(object):
    """
        Tensorflow based implementation of TransE method
        User train method to update the embeddings.
    """

    def __init__(self, graph_er, dimension=10, batchSize=100,
                    learning_rate=0.1, margin=1, regularizer_scale = 0.1, neg_rate=1, neg_rel_rate=0):
        logger.info("Begin generating TransH embeddings with dimension : %d" ,dimension)

        self.dimension = dimension #Embedding Dimension
        self.batchSize = batchSize #BatchSize for Stochastic Gradient Decent
        self.learning_rate = learning_rate #Learning rate for optmizer
        self.margin = margin #margin or bias used for loss computation
        self.entity = graph_er.entity #List of entities in Knowledge graph
        self.relation = graph_er.relation #List of relationships in Knowledge Graph
        self.neg_rate = neg_rate #Number of Negative samples to generate by replacing head or tail
        self.neg_rel_rate = neg_rel_rate #Number fo negative samples by replacing realtion

        # List of triples. Remove last incomplete batch if any.
        self.triples = np.array(graph_er.triples[0: (len(graph_er.triples) - len(graph_er.triples)%batchSize)])
        self.ntriples = np.array(get_negative_samples(self.triples, len(self.entity),
                                        len(self.entity), len(self.relation), graph_er.entity_pairs,
                                        neg_rate=neg_rate, neg_rel_rate=neg_rel_rate))
        logger.info("Shape of triples: %s", str(self.triples.shape))
        logger.info("Shape of neg triples: %s", str(self.ntriples.shape))

        #Define Embedding Variables
        initializer = tf.contrib.layers.xavier_initializer(uniform = True)
        regularizer = tf.contrib.layers.l2_regularizer(scale = regularizer_scale)

        self.ent_embeddings = tf.get_variable(name = "ent_embeddings", shape = [len(self.entity), dimension],
                                    initializer = initializer, regularizer = regularizer)
        self.rel_embeddings = tf.get_variable(name = "rel_embeddings", shape = [len(self.relation), dimension],
                                    initializer = initializer, regularizer = regularizer)
        self.norm_vector = tf.get_variable(name = "norm_vector", shape = [len(self.relation), dimension],
                                    initializer = initializer, regularizer = regularizer)

        #Define Placeholders for input
        self.head = tf.placeholder(tf.int32, shape=[self.batchSize, 1])
        self.tail = tf.placeholder(tf.int32, shape=[self.batchSize, 1])
        self.rel = tf.placeholder(tf.int32, shape=[self.batchSize, 1])
        self.neg_head = tf.placeholder(tf.int32, shape=[self.batchSize, (neg_rate + neg_rel_rate)])
        self.neg_tail= tf.placeholder(tf.int32, shape=[self.batchSize, (neg_rate + neg_rel_rate)])
        self.neg_rel= tf.placeholder(tf.int32, shape=[self.batchSize, (neg_rate + neg_rel_rate)])

        #Load Embedding Vectors
        pos_h = tf.nn.embedding_lookup(self.ent_embeddings, self.head)
        pos_t = tf.nn.embedding_lookup(self.ent_embeddings, self.tail)
        pos_r = tf.nn.embedding_lookup(self.rel_embeddings, self.rel)
        pos_nh = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_head)
        pos_nt = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_tail)
        pos_nr = tf.nn.embedding_lookup(self.rel_embeddings, self.neg_rel)
        pos_norm = tf.nn.embedding_lookup(self.norm_vector, self.rel)
        pos_nnorm = tf.nn.embedding_lookup(self.norm_vector, self.neg_rel)

        #Normalize embedding vectors
        pos_h = tf.nn.l2_normalize(pos_h, [1,2])
        pos_t = tf.nn.l2_normalize(pos_t, [1,2])
        pos_r = tf.nn.l2_normalize(pos_r, [1,2])

        pos_nh = tf.nn.l2_normalize(pos_nh, [1,2])
        pos_nt = tf.nn.l2_normalize(pos_nt, [1,2])
        pos_nr = tf.nn.l2_normalize(pos_nr, [1,2])

        pos_norm = tf.nn.l2_normalize(pos_norm, [1,2])
        pos_nnorm = tf.nn.l2_normalize(pos_nnorm, [1,2])

        #Project entities to hyperplane
        pos_h = self._transfer(pos_h, pos_norm)
        pos_t = self._transfer(pos_t, pos_norm)
        pos_nh = self._transfer(pos_nh, pos_nnorm)
        pos_nt = self._transfer(pos_nt, pos_nnorm)
        logger.info("Pos Triple Shapes: %s, %s, %s", str(pos_h.shape), str(pos_t.shape), str(pos_r.shape))
        logger.info("Neg Triple Shapes: %s, %s, %s", str(pos_nh.shape), str(pos_nt.shape), str(pos_nr.shape))

        #Compute Loss
        _p_score = self._calc(pos_h, pos_t, pos_r)
        _n_score = self._calc(pos_nh, pos_nt, pos_nr)

        p_score = tf.reduce_sum(tf.reduce_mean(_p_score, 0, keepdims=False), keepdims=True, axis=1)
        n_score = tf.reduce_sum(tf.reduce_mean(_n_score, 0, keepdims=False), keepdims=True, axis=1)
        logger.info("PScore Shape %s. N_score Shape: %s", str(p_score.shape), str(n_score.shape))

        self.loss = tf.reduce_sum(tf.maximum(p_score - n_score + self.margin, 0))

        #collect summary parameters
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('pos_score', tf.reduce_mean(p_score))
        tf.summary.scalar('neg_score', tf.reduce_mean(n_score))

        #Configure optimizer
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

        #Configure session
        self.sess = tf.Session()

        #Confirgure summary location
        self.merged = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(get_tf_summary_file_path(logger), self.sess.graph)

        #Configure Saver
        self.saver = tf.train.Saver()

    def _calc(self, h, t, r):
        """
            TransE objective function.
            It estimates embeddings as translation from head to tail entity.
        """
        return abs(h + r - t)

    def _transfer(self, e, n):
        return e - tf.reduce_sum(e * n, 1, keep_dims = True) * n

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
                    neg_batchend = min(len(self.ntriples), i + self.batchSize*(self.neg_rate + self.neg_rel_rate))
                    feed_dict = {
                        self.head : self.triples[i:batchend][:,0].reshape(self.batchSize, 1),
                        self.tail : self.triples[i:batchend][:,1].reshape(self.batchSize, 1),
                        self.rel  : self.triples[i:batchend][:,2].reshape(self.batchSize, 1),
                        self.neg_head : self.ntriples[i:neg_batchend][:,0].reshape(self.batchSize, (self.neg_rate + self.neg_rel_rate)),
                        self.neg_tail : self.ntriples[i:neg_batchend][:,1].reshape(self.batchSize, (self.neg_rate + self.neg_rel_rate)),
                        self.neg_rel : self.ntriples[i:neg_batchend][:,2].reshape(self.batchSize, (self.neg_rate + self.neg_rel_rate))
                        }

                    if batchend == len(self.triples):
                        _ , cur_loss, summary = self.sess.run([self.optimizer, self.loss, self.merged],
                                                            feed_dict=feed_dict)
                        self.summary_writer.add_summary(summary, epoch)
                    else:
                        _, cur_loss = self.sess.run([self.optimizer, self.loss], feed_dict=feed_dict)

                    loss = loss + cur_loss

                if loss:
                    logger.info("Epoch: %d Loss: %f", epoch, loss)
                else:
                    logger.info("Zero Loss, finish training in %d epochs", epoch)
                    break
        return loss

    def get_ent_embeddings(self):
        with self.sess.as_default():
            return tf.nn.embedding_lookup(self.ent_embeddings, range(0, len(self.entity))).eval()

    def save_model(self, filename):
        with self.sess.as_default():
            self.saver.save(self.sess, filename)
        logger.info("Saved tf train model with filename: %s", filename)
        return True

    def restore_model(self, filename):
        with self.sess.as_default():
            self.saver.restore(self.sess, filename)
        logger.info("Restored tf train model from filename: %s", filename)
        return True

    def close_tf_session(self):
        tf.reset_default_graph()
        self.sess.close()

    def __str__(self):
        return "TransH"

    def __del__(self):
        self.close_tf_session()

