import tensorflow as tf
import numpy as np

from common import sigmoid, get_logger
from scipy import spatial
import timeit

logger = get_logger('RL.ERER.MTransE')

class MTransE(object):
    """
        Tensorflow based implementation of MTransE method
        Use train method to update the embeddings.
    """

    def __init__(self, graph_erer, dimension=10, batchSize=100, learning_rate=0.1,
                                    regularizer_scale = 0.1, alpha=5):
        logger.info("Begin generating MTransE embeddings with dimension : %d" ,dimension)

        self.dimension = dimension #Embedding Dimension
        self.batchSize = batchSize #BatchSize for Stochastic Gradient Decent
        self.learning_rate = learning_rate #Learning rate for optmizer
        self.entityA = graph_erer.entityA #List of entities in Knowledge graph A
        self.relationA = graph_erer.relationA #List of relationships in Knowledge Graph A
        self.entityB = graph_erer.entityB #List of entities in Knowledge graph A
        self.relationB = graph_erer.relationB #List of relationships in Knowledge Graph A
        self.alpha = alpha #Hyperparmeter to weight knowledge model vs alignment model
        #Todo: use alpha

        # List of triples. Remove last incomplete batch if any.
        self.triplesA = np.array(graph_erer.triplesA[0: (len(graph_erer.triplesA) -
                                        len(graph_erer.triplesA)%batchSize)])
        self.triplesB = np.array(graph_erer.triplesB[0: (len(graph_erer.triplesB) -
                                        len(graph_erer.triplesB)%batchSize)])
        logger.info("Shape of triples A: %s", str(self.triplesA.shape))
        logger.info("Shape of triples B: %s", str(self.triplesB.shape))

        # List of ILLs / linked entities. Remove last incomplete batch if any.
        self.prior_pairs = np.array(graph_erer.prior_pairs[0: (len(graph_erer.prior_pairs) -
                                        len(graph_erer.prior_pairs)%batchSize)])
        logger.info("Shape of prior_pairs: %s", str(self.prior_pairs.shape))

        #Define Embedding Variables
        initializer = tf.contrib.layers.xavier_initializer(uniform = False)
        regularizer = tf.contrib.layers.l2_regularizer(scale = regularizer_scale)
        self.ent_embeddings_A = tf.get_variable(name = "ent_embeddings_A", shape = [len(self.entityA), dimension],
                                    initializer = initializer, regularizer = regularizer)
        self.ent_embeddings_B = tf.get_variable(name = "ent_embeddings_B", shape = [len(self.entityB), dimension],
                                    initializer = initializer, regularizer = regularizer)
        self.rel_embeddings_A = tf.get_variable(name = "rel_embeddings_A", shape = [len(self.relationA), dimension],
                                    initializer = initializer, regularizer = regularizer)
        self.rel_embeddings_B = tf.get_variable(name = "rel_embeddings_B", shape = [len(self.relationB), dimension],
                                    initializer = initializer, regularizer = regularizer)
        self.ent_translation = tf.get_variable(name = "ent_translation", shape = [dimension, dimension],
                                    initializer = initializer, regularizer = regularizer)

        #Define Placeholders for input
        self.headA = tf.placeholder(tf.int32, shape=[self.batchSize])
        self.tailA = tf.placeholder(tf.int32, shape=[self.batchSize])
        self.relA = tf.placeholder(tf.int32, shape=[self.batchSize])
        self.headB = tf.placeholder(tf.int32, shape=[self.batchSize])
        self.tailB = tf.placeholder(tf.int32, shape=[self.batchSize])
        self.relB = tf.placeholder(tf.int32, shape=[self.batchSize])

        self.ent_A = tf.placeholder(tf.int32, shape=[self.batchSize])
        self.ent_B = tf.placeholder(tf.int32, shape=[self.batchSize])

        pos_ha = tf.nn.embedding_lookup(self.ent_embeddings_A, self.headA)
        pos_ta = tf.nn.embedding_lookup(self.ent_embeddings_A, self.tailA)
        pos_ra = tf.nn.embedding_lookup(self.rel_embeddings_A, self.relA)
        pos_hb = tf.nn.embedding_lookup(self.ent_embeddings_B, self.headB)
        pos_tb = tf.nn.embedding_lookup(self.ent_embeddings_B, self.tailB)
        pos_rb = tf.nn.embedding_lookup(self.rel_embeddings_B, self.relB)

        pos_entA = tf.nn.embedding_lookup(self.ent_embeddings_A, self.ent_A)
        pos_entB = tf.nn.embedding_lookup(self.ent_embeddings_B, self.ent_B)
        pos_ent_trans = tf.nn.embedding_lookup(self.ent_translation, range(0, dimension))

        #Normalize embeddings
        pos_ha = tf.nn.l2_normalize(pos_ha, 1)
        pos_ta = tf.nn.l2_normalize(pos_ta, 1)
        pos_ra = tf.nn.l2_normalize(pos_ra, 1)
        pos_hb = tf.nn.l2_normalize(pos_hb, 1)
        pos_tb = tf.nn.l2_normalize(pos_tb, 1)
        pos_rb = tf.nn.l2_normalize(pos_rb, 1)

        pos_entA = tf.nn.l2_normalize(pos_entA, 1)
        pos_entB = tf.nn.l2_normalize(pos_entB, 1)
        pos_ent_trans = tf.nn.l2_normalize(pos_ent_trans, 1)

        logger.info("Triple Shapes A: %s, %s, %s", str(pos_ha.shape), str(pos_ta.shape), str(pos_ra.shape))
        logger.info("Triple Shapes B: %s, %s, %s", str(pos_hb.shape), str(pos_tb.shape), str(pos_rb.shape))
        logger.info("Prior Pairs Shape: %s, %s, %s", str(pos_entA.shape), str(pos_entB.shape), str(pos_ent_trans.shape))

        #Compute loss: knowledge model
        _a_score = self._calc(pos_ha, pos_ta, pos_ra)
        #Todo: maybe use: tf.abs(tf.subtract(tf.add(pos_ha, pos_ra), post_ta))
        _b_score = self._calc(pos_hb, pos_tb, pos_rb)

        self.a_score = tf.reduce_mean(tf.reduce_sum(_a_score, keepdims=True, axis=1))
        self.b_score = tf.reduce_mean(tf.reduce_sum(_b_score, keepdims=True, axis=1))
        logger.info("AScore Shape %s. Bscore Shape: %s", str(self.a_score.shape), str(self.b_score.shape))

        self.optimizer_A = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(
                                self.a_score, var_list=[self.ent_embeddings_A, self.rel_embeddings_A])
        self.optimizer_B = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(
                                self.b_score, var_list=[self.ent_embeddings_B, self.rel_embeddings_B])

        #Compute loss: alignment model
        _alignment_loss = tf.abs(tf.subtract(tf.matmul(pos_entA, pos_ent_trans), pos_entB))
        self.alignment_loss = tf.reduce_mean(tf.reduce_sum(_alignment_loss, keepdims=True, axis=1))
        logger.info("Alignment loss Shape %s.", str(self.alignment_loss.shape))

        #Configure optimizer
        self.optimizer_AM = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(
                                self.alignment_loss, var_list=[self.ent_embeddings_A,
                                self.ent_embeddings_B, self.ent_translation])

        #Configure session
        self.sess = tf.Session()

    def _calc(self, h, t, r):
        """
            TransE objective function.
            It estimates embeddings as translation from head to tail entity.
        """
        return abs(h + r - t)

    def train(self, max_epochs=100):
        """
            Method to train the model and update the embeddings.
        """
        loss = 0
        with self.sess.as_default():
            self.sess.run(tf.global_variables_initializer())

            for epoch in range(0, max_epochs):
                loss = 0
                #Learning embeddings for Knowledge Graph A
                a_loss = 0
                for i in np.arange(0, len(self.triplesA), self.batchSize):
                    batchend = min(len(self.triplesA), i + self.batchSize)
                    feed_dict = {
                        self.headA : self.triplesA[i:batchend][:, 0],
                        self.tailA : self.triplesA[i:batchend][:, 1],
                        self.relA  : self.triplesA[i:batchend][:, 2],
                        }
                    _ , cur_loss = self.sess.run([self.optimizer_A, self.a_score], feed_dict=feed_dict)
                    a_loss = a_loss + cur_loss

                #Learning embeddings for Knowledge Graph B
                b_loss = 0
                for i in np.arange(0, len(self.triplesB), self.batchSize):
                    batchend = min(len(self.triplesB), i + self.batchSize)
                    feed_dict = {
                        self.headB : self.triplesB[i:batchend][:, 0],
                        self.tailB : self.triplesB[i:batchend][:, 1],
                        self.relB  : self.triplesB[i:batchend][:, 2],
                        }
                    _ , cur_loss = self.sess.run([self.optimizer_B, self.b_score], feed_dict=feed_dict)
                    b_loss = b_loss + cur_loss

                #Learning alignment model
                am_loss = 0
                for i in np.arange(0, len(self.prior_pairs), self.batchSize):
                    batchend = min(len(self.prior_pairs), i + self.batchSize)
                    feed_dict = {
                        self.ent_A : self.prior_pairs[i:batchend][:, 0],
                        self.ent_B : self.prior_pairs[i:batchend][:, 1],
                        }
                    _ , cur_loss = self.sess.run([self.optimizer_AM, self.alignment_loss], feed_dict=feed_dict)
                    am_loss = am_loss + cur_loss

                loss = a_loss + b_loss + am_loss
                if loss:
                    logger.info("Epoch: %d Loss: %f", epoch, loss)
                else:
                    logger.info("Zero Loss, finish training in %d epochs", epoch)
                    break
        return loss

    def get_ent_embeddings_A(self):
        with self.sess.as_default():
            return tf.nn.embedding_lookup(self.ent_embeddings_A, range(0, len(self.entityA))).eval()

    def get_ent_embeddings_B(self):
        with self.sess.as_default():
            return tf.nn.embedding_lookup(self.ent_embeddings_B, range(0, len(self.entityB))).eval()

    def close_tf_session(self):
        tf.reset_default_graph()
        self.sess.close()

    def __str__(self):
        return "TransE"

    def __del__(self):
        self.close_tf_session()
