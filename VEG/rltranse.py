import config
import tensorflow as tf
import numpy as np

from common import get_logger, get_negative_samples, get_tf_summary_file_path
from scipy import spatial
import timeit

logger = get_logger('RL.VEG.RLTransE')

class RLTransE(object):
    """
        Tensorflow based implementation of TransE method
        Use train method to update the embeddings.
    """

    def __init__(self, graph_veg, dimension=10, batchSize=100, learning_rate=0.1, margin=1,
                                    regularizer_scale = 0.1, neg_rate=1, neg_rel_rate=0):
        logger.info("Begin generating RLTransE embeddings with dimension : %d" ,dimension)

        self.dimension = dimension #Embedding Dimension
        self.batchSize = batchSize #BatchSize for Stochastic Gradient Decent
        self.learning_rate = learning_rate #Learning rate for optmizer
        self.margin = margin #margin or bias used for loss computation
        self.relation_value_map = graph_veg.relation_value_map #List of entities in Knowledge graph
        self.relation = graph_veg.relation #List of relationships in Knowledge Graph
        self.neg_rate = neg_rate #Number of Negative samples to generate by replacing head or tail
        self.neg_rel_rate = neg_rel_rate #Number fo negative samples by replacing realtion

        # List of triples. Remove last incomplete batch if any.
        self.triples = np.array(graph_veg.train_triples[0: (len(graph_veg.train_triples) -
                                        len(graph_veg.train_triples)%batchSize)])
        self.ntriples = []
        for index in range(len(self.relation)):
            rel_triples = [(h, t, r) for (h, t, r) in self.triples if r == index]
            val_count = len(self.relation_value_map[self.relation[index]])
            self.ntriples.extend(get_negative_samples(rel_triples, val_count, val_count,
                                        len(self.relation), [], neg_rate=neg_rate, neg_rel_rate=neg_rel_rate))
        self.ntriples = np.array(self.ntriples)
        logger.info("Shape of triples: %s", str(self.triples.shape))
        logger.info("Shape of neg triples: %s", str(self.ntriples.shape))

        self.val_triples = np.array(graph_veg.val_triples[0: (len(graph_veg.val_triples) -
                                        len(graph_veg.val_triples)%batchSize)])
        self.val_ntriples = []
        for index in range(len(self.relation)):
            rel_triples = [(h, t, r) for (h, t, r) in self.val_triples if r == index]
            value_count = len(self.relation_value_map[self.relation[index]])
            self.val_ntriples.extend(get_negative_samples(rel_triples, value_count, value_count,
                        len(self.relation), [], neg_rate=neg_rate, neg_rel_rate=neg_rel_rate))
        self.val_ntriples = np.array(self.val_ntriples)
        logger.info("Shape of val triples: %s", str(self.val_triples.shape))
        logger.info("Shape of val neg triples: %s", str(self.val_ntriples.shape))

        #Define Embedding Variables
        initializer = tf.contrib.layers.xavier_initializer(uniform = True)
        regularizer = tf.contrib.layers.l2_regularizer(scale = regularizer_scale)

        self.max_val_count = 0
        for index in range(len(self.relation)):
            val_count = len(self.relation_value_map[self.relation[index]])
            if val_count > self.max_val_count:
                self.max_val_count = val_count

        self.val_embeddings = tf.get_variable(name = "val_embeddings",
                                    shape = [len(self.relation) * self.max_val_count, dimension],
                                    initializer = initializer, regularizer = regularizer)
        self.rel_embeddings = tf.get_variable(name = "rel_embeddings",
                                    shape = [len(self.relation), dimension],
                                    initializer = initializer, regularizer = regularizer)

        #Define Placeholders for input
        self.head = tf.placeholder(tf.int32, shape=[self.batchSize, 1])
        self.tail = tf.placeholder(tf.int32, shape=[self.batchSize, 1])
        self.rel = tf.placeholder(tf.int32, shape=[self.batchSize, 1])
        self.neg_head = tf.placeholder(tf.int32, shape=[self.batchSize, (neg_rate + neg_rel_rate)])
        self.neg_tail = tf.placeholder(tf.int32, shape=[self.batchSize, (neg_rate + neg_rel_rate)])
        self.neg_rel = tf.placeholder(tf.int32, shape=[self.batchSize, (neg_rate + neg_rel_rate)])

        pos_h = tf.nn.embedding_lookup(self.val_embeddings, self.head + self.rel * self.max_val_count)
        pos_t = tf.nn.embedding_lookup(self.val_embeddings, self.tail + self.rel * self.max_val_count)
        pos_r = tf.nn.embedding_lookup(self.rel_embeddings, self.rel)
        pos_nh = tf.nn.embedding_lookup(self.val_embeddings, self.neg_head + self.neg_rel * self.max_val_count)
        pos_nt = tf.nn.embedding_lookup(self.val_embeddings, self.neg_tail + self.neg_rel * self.max_val_count)
        pos_nr = tf.nn.embedding_lookup(self.rel_embeddings, self.neg_rel)

        #Normalize embeddings
        pos_h = tf.nn.l2_normalize(pos_h, [1,2])
        pos_t = tf.nn.l2_normalize(pos_t, [1,2])
        pos_r = tf.nn.l2_normalize(pos_r, [1,2])
        pos_nh = tf.nn.l2_normalize(pos_nh, [1,2])
        pos_nt = tf.nn.l2_normalize(pos_nt, [1,2])
        pos_nr = tf.nn.l2_normalize(pos_nr, [1,2])

        logger.info("Pos Triple Shapes: %s, %s, %s", str(pos_h.shape), str(pos_t.shape), str(pos_r.shape))
        logger.info("Neg Triple Shapes: %s, %s, %s", str(pos_nh.shape), str(pos_nt.shape), str(pos_nr.shape))

        #Compute loss
        _p_score = self._calc(pos_h, pos_t, pos_r)
        _n_score = self._calc(pos_nh, pos_nt, pos_nr)

        p_score =  tf.reduce_sum(tf.reduce_mean(_p_score, 1, keepdims = False), 1, keepdims = True)
        n_score =  tf.reduce_sum(tf.reduce_mean(_n_score, 1, keepdims = False), 1, keepdims = True)

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
        self.summary_writer = tf.summary.FileWriter(get_tf_summary_file_path(logger) + '/train', self.sess.graph)
        self.validation_summary_writer = tf.summary.FileWriter(get_tf_summary_file_path(logger) + '/val',
                                    self.sess.graph)


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
                batch_starts = np.arange(0, len(self.triples), self.batchSize)
                np.random.shuffle(batch_starts)
                for i in batch_starts:
                    batchend = min(len(self.triples), i + self.batchSize)
                    neg_batchend = min(len(self.ntriples), i + self.batchSize*(self.neg_rate + self.neg_rel_rate))
                    feed_dict = {
                        self.head : self.triples[i:batchend][:, 0].reshape(self.batchSize, 1),
                        self.tail : self.triples[i:batchend][:, 1].reshape(self.batchSize, 1),
                        self.rel  : self.triples[i:batchend][:, 2].reshape(self.batchSize, 1),
                        self.neg_head : self.ntriples[i:neg_batchend][:, 0].reshape(self.batchSize, (self.neg_rate + self.neg_rel_rate)),
                        self.neg_tail : self.ntriples[i:neg_batchend][:, 1].reshape(self.batchSize, (self.neg_rate + self.neg_rel_rate)),
                        self.neg_rel : self.ntriples[i:neg_batchend][:, 2].reshape(self.batchSize, (self.neg_rate + self.neg_rel_rate))
                        }

                    if batchend == len(self.triples):
                        _, cur_loss, summary = self.sess.run([self.optimizer, self.loss, self.merged], feed_dict=feed_dict)
                        self.summary_writer.add_summary(summary, epoch)
                    else:
                        #train network on batch
                        _, cur_loss = self.sess.run([self.optimizer, self.loss], feed_dict=feed_dict)

                    if type(cur_loss) == list:
                        cur_loss = cur_loss[0]
                    loss = loss + cur_loss
                if loss:
                    logger.info("Epoch: %d Loss: %f", epoch, loss)
                else:
                    logger.info("Zero Loss, finish training in %d epochs", epoch)
                    break

                val_loss = 0
                batch_starts = np.arange(0, len(self.val_triples), self.batchSize)
                np.random.shuffle(batch_starts)
                for i in batch_starts:
                    batchend = min(len(self.val_triples), i + self.batchSize)
                    neg_batchend = min(len(self.val_ntriples), i + self.batchSize*(self.neg_rate + self.neg_rel_rate))
                    feed_dict = {
                        self.head : self.val_triples[i:batchend][:, 0].reshape(self.batchSize, 1),
                        self.tail : self.val_triples[i:batchend][:, 1].reshape(self.batchSize, 1),
                        self.rel  : self.val_triples[i:batchend][:, 2].reshape(self.batchSize, 1),
                        self.neg_head : self.val_ntriples[i:neg_batchend][:, 0].reshape(self.batchSize, (self.neg_rate + self.neg_rel_rate)),
                        self.neg_tail : self.val_ntriples[i:neg_batchend][:, 1].reshape(self.batchSize, (self.neg_rate + self.neg_rel_rate)),
                        self.neg_rel : self.val_ntriples[i:neg_batchend][:, 2].reshape(self.batchSize, (self.neg_rate + self.neg_rel_rate))
                        }

                    if batchend == len(self.triples):
                        cur_loss, summary = self.sess.run([self.loss, self.merged], feed_dict=feed_dict)
                        self.validation_summary_writer.add_summary(summary, epoch)
                    else:
                        #validate network on batch
                        cur_loss = self.sess.run([self.loss], feed_dict=feed_dict)

                    if type(cur_loss) == list:
                        cur_loss = cur_loss[0]
                    val_loss = val_loss + cur_loss
                if val_loss:
                    logger.info("Epoch: %d VAL Loss: %f", epoch, val_loss)
                else:
                    logger.info("Zero VAL Loss, finish training in %d epochs", epoch)
                    break
        return (loss, val_loss)

    def get_val_embeddings(self):
        embeddings = {}
        with self.sess.as_default():
            for index in range(len(self.relation)):
                start_index = index * self.max_val_count
                embeddings[self.relation[index]] = tf.nn.embedding_lookup(self.val_embeddings, \
                            range(start_index, start_index + self.max_val_count)).eval()
        return embeddings

    def get_rel_embeddings(self):
        with self.sess.as_default():
            return tf.nn.embedding_lookup(self.rel_embeddings, range(0, len(self.relation))).eval()

    def close_tf_session(self):
        tf.reset_default_graph()
        self.sess.close()

    def __str__(self):
        return "RLTransE"

    def __del__(self):
        self.close_tf_session()
