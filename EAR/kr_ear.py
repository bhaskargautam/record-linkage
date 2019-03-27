import numpy as np
import tensorflow as tf

from common import get_logger, get_negative_samples, get_tf_summary_file_path
from scipy import spatial

logger = get_logger('RL.EAR.KR_EAR')

class KR_EAR(object):

    def __init__(self, graph_ear, dimension=10, learning_rate=0.1, batchSize=100, margin=1,
                                regularizer_scale = 0.1, neg_rate=1, neg_rel_rate=0):
        self.entity = graph_ear.entity
        self.attribute = graph_ear.attribute
        self.relation = graph_ear.relation
        self.value = graph_ear.value
        self.dimension = dimension
        self.learning_rate = learning_rate
        self.batchSize = batchSize
        self.margin = margin
        self.neg_rate = neg_rate
        self.neg_rel_rate = neg_rel_rate

        # List of triples. Remove last incomplete batch if any.
        self.atriples = np.array(graph_ear.atriples[0: (len(graph_ear.atriples) - len(graph_ear.atriples)%batchSize)])
        self.rtriples = np.array(graph_ear.rtriples[0: (len(graph_ear.rtriples) - len(graph_ear.rtriples)%batchSize)])
        logger.info("Modified Atriples size: %d", len(self.atriples))
        logger.info("Modified Rtriples size: %d", len(self.rtriples))

        #Collect Negative Samples
        self.nrtriples = np.array(get_negative_samples(self.rtriples, len(self.entity),
                                    len(self.entity), len(self.relation), graph_ear.entity_pairs,
                                    neg_rate = self.neg_rate, neg_rel_rate = self.neg_rel_rate))
        self.natriples = np.array(get_negative_samples(self.atriples, len(self.entity),
                                len(self.value), len(self.attribute), [],
                                neg_rate = self.neg_rate, neg_rel_rate = self.neg_rel_rate))

        #Define Embedding Variables
        initializer = tf.contrib.layers.xavier_initializer(uniform = True)
        regularizer = tf.contrib.layers.l2_regularizer(scale = regularizer_scale)

        self.ent_embeddings = tf.get_variable(name = "ent_embeddings", shape = [len(self.entity), dimension],
                                    initializer = initializer, regularizer = regularizer)
        self.rel_embeddings = tf.get_variable(name = "rel_embeddings", shape = [len(self.relation), dimension],
                                    initializer = initializer, regularizer = regularizer)
        self.attr_embeddings = tf.get_variable(name = "attr_embeddings", shape = [len(self.attribute), dimension],
                                    initializer = initializer, regularizer = regularizer)
        self.val_embeddings = tf.get_variable(name = "val_embeddings", shape = [len(self.value), dimension],
                                    initializer = initializer, regularizer = regularizer)
        self.projection_matrix = tf.get_variable(name = "projection_matrix", shape = [len(self.attribute), dimension],
                                    initializer = initializer, regularizer = regularizer)

        #Define Placeholders for input
        self.head = tf.placeholder(tf.int32, shape=[self.batchSize, 1])
        self.tail = tf.placeholder(tf.int32, shape=[self.batchSize, 1])
        self.rel = tf.placeholder(tf.int32, shape=[self.batchSize, 1])
        self.neg_head = tf.placeholder(tf.int32, shape=[self.batchSize, (self.neg_rel_rate + self.neg_rate)])
        self.neg_tail = tf.placeholder(tf.int32, shape=[self.batchSize, (self.neg_rel_rate + self.neg_rate)])
        self.neg_rel = tf.placeholder(tf.int32, shape=[self.batchSize, (self.neg_rel_rate + self.neg_rate)])

        self.attr_head = tf.placeholder(tf.int32, shape=[self.batchSize, 1])
        self.val = tf.placeholder(tf.int32, shape=[self.batchSize, 1])
        self.attr = tf.placeholder(tf.int32, shape=[self.batchSize, 1])
        self.neg_attr_head = tf.placeholder(tf.int32, shape=[self.batchSize, (self.neg_rel_rate + self.neg_rate)])
        self.neg_val = tf.placeholder(tf.int32, shape=[self.batchSize, (self.neg_rel_rate + self.neg_rate)])
        self.neg_attr = tf.placeholder(tf.int32, shape=[self.batchSize, (self.neg_rel_rate + self.neg_rate)])

        #Load Embedding Vectors for Relational Triple
        pos_h = tf.nn.embedding_lookup(self.ent_embeddings, self.head)
        pos_t = tf.nn.embedding_lookup(self.ent_embeddings, self.tail)
        pos_r = tf.nn.embedding_lookup(self.rel_embeddings, self.rel)
        pos_nh = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_head)
        pos_nt = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_tail)
        pos_nr = tf.nn.embedding_lookup(self.rel_embeddings, self.neg_rel)

        #Normalize vectors
        pos_h = tf.nn.l2_normalize(pos_h, [1,2])
        pos_t = tf.nn.l2_normalize(pos_t, [1,2])
        pos_r = tf.nn.l2_normalize(pos_r, [1,2])
        pos_nh = tf.nn.l2_normalize(pos_nh, [1,2])
        pos_nt = tf.nn.l2_normalize(pos_nt, [1,2])
        pos_nr = tf.nn.l2_normalize(pos_nr, [1,2])

        #Load Embedding Vectors for Attributional Triple
        pos_attr_h = tf.nn.embedding_lookup(self.ent_embeddings, self.attr_head)
        pos_val = tf.nn.embedding_lookup(self.val_embeddings, self.val)
        pos_attr = tf.nn.embedding_lookup(self.attr_embeddings, self.attr)
        pos_attr_nh = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_attr_head)
        pos_attr_nv = tf.nn.embedding_lookup(self.val_embeddings, self.neg_val)
        pos_attr_na = tf.nn.embedding_lookup(self.attr_embeddings, self.neg_attr)

        #Normalize vectors
        pos_attr_h = tf.nn.l2_normalize(pos_attr_h, [1,2])
        pos_val = tf.nn.l2_normalize(pos_val, [1,2])
        pos_attr = tf.nn.l2_normalize(pos_attr, [1,2])
        pos_attr_nh = tf.nn.l2_normalize(pos_attr_nh, [1,2])
        pos_attr_nv = tf.nn.l2_normalize(pos_attr_nv, [1,2])
        pos_attr_na = tf.nn.l2_normalize(pos_attr_na, [1,2])

        #Load Normal Vectors
        pos_proj = tf.nn.embedding_lookup(self.projection_matrix, self.attr)
        pos_nproj = tf.nn.embedding_lookup(self.projection_matrix, self.neg_attr)

        #Normalize vectors
        pos_proj = tf.nn.l2_normalize(pos_proj, [1,2])
        pos_nproj = tf.nn.l2_normalize(pos_nproj, [1,2])

        proj_pos_attr_h = self._transfer(pos_attr_h, pos_proj)
        proj_pos_attr_nh = self._transfer(pos_attr_nh, pos_nproj)

        #Compute Loss
        _p_score = self._calc(pos_h, pos_t, pos_r)
        _n_score = self._calc(pos_nh, pos_nt, pos_nr)

        _ap_score = self._attr_calc(proj_pos_attr_h, pos_val, pos_attr)
        _an_score = self._attr_calc(proj_pos_attr_nh, pos_attr_nv, pos_attr_na)

        p_score = tf.reduce_sum(tf.reduce_mean(_p_score, 1, keepdims=False), axis=1, keepdims=True)
        n_score = tf.reduce_sum(tf.reduce_mean(_n_score, 1, keepdims=False), axis=1, keepdims=True)
        ap_score = tf.reduce_sum(tf.reduce_mean(_ap_score, 1, keepdims=False), axis=1, keepdims=True)
        an_score = tf.reduce_sum(tf.reduce_mean(_an_score, 1, keepdims=False), axis=1, keepdims=True)
        self.rel_loss = tf.reduce_sum(tf.maximum(p_score - n_score + self.margin, 0))
        self.attr_loss = tf.reduce_sum(tf.maximum(ap_score - an_score + self.margin, 0))

        #Configure optimizer
        self.rel_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.rel_loss)
        self.attr_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.attr_loss)

         #Configure session
        self.sess = tf.Session()

        #Collect summary for tensorboard
        tf.summary.scalar('attr_loss', self.attr_loss, collections=['attr'])
        tf.summary.scalar('rel_loss', self.rel_loss, collections=['rel'])
        tf.summary.scalar('p_score', tf.reduce_mean(p_score), collections=['rel'])
        tf.summary.scalar('n_score', tf.reduce_mean(n_score), collections=['rel'])
        tf.summary.scalar('ap_score', tf.reduce_mean(ap_score), collections=['attr'])
        tf.summary.scalar('an_score', tf.reduce_mean(an_score), collections=['attr'])


        #Confirgure summary location
        self.merged_attr = tf.summary.merge_all(key='attr')
        self.merged_rel = tf.summary.merge_all(key='rel')
        self.attr_summary_writer = tf.summary.FileWriter(get_tf_summary_file_path(logger) + '_attr', self.sess.graph)
        self.rel_summary_writer = tf.summary.FileWriter(get_tf_summary_file_path(logger) +'_rel', self.sess.graph)


    def _calc(self, h, t, r):
        """
            TransE objective function.
            It estimates embeddings as translation from head to tail entity.
        """
        return abs(h + r - t)

    def _attr_calc(self, h, v, a):
        return abs(tf.nn.tanh(h + a) - v)

    def _transfer(self, e, n):
        return e - tf.reduce_sum(e * n, 1, keepdims=True) * n

    def train(self, max_epochs=100):
        loss = 0
        with self.sess.as_default():
            self.sess.run(tf.global_variables_initializer())
            for epoch in range(0, max_epochs):
                rel_loss = attr_loss = 0

                #Relation Triple Encoder
                for i in np.arange(0, len(self.rtriples), self.batchSize):
                    batchend = min(len(self.rtriples), i + self.batchSize)
                    neg_batchend = min(len(self.nrtriples), i + self.batchSize*(self.neg_rate + self.neg_rel_rate))
                    feed_dict = {
                        self.head : self.rtriples[i:batchend][:,0].reshape(self.batchSize, 1),
                        self.tail : self.rtriples[i:batchend][:,1].reshape(self.batchSize, 1),
                        self.rel : self.rtriples[i:batchend][:,2].reshape(self.batchSize, 1),
                        self.neg_head : self.nrtriples[i:neg_batchend][:,0].reshape(self.batchSize, (self.neg_rate + self.neg_rel_rate)),
                        self.neg_tail : self.nrtriples[i:neg_batchend][:,1].reshape(self.batchSize, (self.neg_rate + self.neg_rel_rate)),
                        self.neg_rel : self.nrtriples[i:neg_batchend][:,2].reshape(self.batchSize, (self.neg_rate + self.neg_rel_rate))
                    }

                    if batchend == len(self.rtriples):
                        _, cur_rel_loss, summary = self.sess.run([self.rel_optimizer, self.rel_loss, self.merged_rel],
                            feed_dict=feed_dict)
                        self.rel_summary_writer.add_summary(summary, epoch)
                    else:
                        _, cur_rel_loss = self.sess.run([self.rel_optimizer, self.rel_loss],
                            feed_dict=feed_dict)

                    #logger.info("Cur rel loss: %f", cur_rel_loss)
                    rel_loss = rel_loss + cur_rel_loss


                #Attributional Triple Encoder
                for i in np.arange(0, len(self.atriples), self.batchSize):
                    batchend = min(len(self.atriples), i + self.batchSize)
                    neg_batchend = min(len(self.natriples), i + self.batchSize*(self.neg_rate + self.neg_rel_rate))

                    feed_dict = {
                        self.attr_head : self.atriples[i:batchend][:,0].reshape(self.batchSize, 1),
                        self.val : self.atriples[i:batchend][:,1].reshape(self.batchSize, 1),
                        self.attr : self.atriples[i:batchend][:,2].reshape(self.batchSize, 1),
                        self.neg_attr_head : self.natriples[i:neg_batchend][:,0].reshape(self.batchSize, (self.neg_rate + self.neg_rel_rate)),
                        self.neg_val : self.natriples[i:neg_batchend][:,1].reshape(self.batchSize, (self.neg_rate + self.neg_rel_rate)),
                        self.neg_attr : self.natriples[i:neg_batchend][:,2].reshape(self.batchSize, (self.neg_rate + self.neg_rel_rate))
                    }
                    if batchend == len(self.atriples):
                        _, cur_attr_loss, summary = self.sess.run([self.attr_optimizer, self.attr_loss, self.merged_attr],
                            feed_dict=feed_dict)
                        self.attr_summary_writer.add_summary(summary, epoch)
                    else:
                        _, cur_attr_loss = self.sess.run([self.attr_optimizer, self.attr_loss],
                            feed_dict=feed_dict)
                    #logger.info("Cur attr loss: %f", cur_attr_loss)
                    attr_loss = attr_loss + cur_attr_loss

                loss = rel_loss + attr_loss

                if loss:
                    logger.info("Epoch: %d Loss: %f Rel_loss: %f, Attr_loss: %f", epoch, loss, rel_loss, attr_loss)
                else:
                    logger.info("Zero Loss, finish training in %d epochs", epoch)
                    break

        return loss

    def get_ent_embeddings(self):
        with self.sess.as_default():
            return tf.nn.embedding_lookup(self.ent_embeddings, range(0, len(self.entity))).eval()

    def close_tf_session(self):
        tf.reset_default_graph()
        self.sess.close()

    def __del__(self):
        self.close_tf_session()

