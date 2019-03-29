import tensorflow as tf
import numpy as np

from common import get_logger, get_negative_samples, get_tf_summary_file_path
from scipy import spatial
import timeit

logger = get_logger('RL.ERER.MTransE')

class MTransE(object):
    """
        Tensorflow based implementation of MTransE method
        Use 'train' method to update the embeddings.
    """

    def __init__(self, graph_erer, dimension=10, batchSize=100, learning_rate=0.1, margin=1,
                        regularizer_scale = 0.1, alpha=5, neg_rate=7, neg_rel_rate=1):
        logger.info("Begin generating MTransE embeddings with dimension : %d" ,dimension)

        self.dimension = dimension #Embedding Dimension
        self.batchSize = batchSize #BatchSize for Stochastic Gradient Decent
        self.learning_rate = learning_rate #Learning rate for optmizer
        self.entityA = graph_erer.entityA #List of entities in Knowledge graph A
        self.relationA = graph_erer.relationA #List of relationships in Knowledge Graph A
        self.entityB = graph_erer.entityB #List of entities in Knowledge graph B
        self.relationB = graph_erer.relationB #List of relationships in Knowledge Graph B
        self.margin = margin
        self.alpha = alpha #Hyperparmeter to weight knowledge model vs alignment model
        self.neg_rel_rate = neg_rel_rate
        self.neg_rate = neg_rate

        # List of triples. Remove last incomplete batch if any.
        self.triplesA = np.array(graph_erer.triplesA[0: (len(graph_erer.triplesA) -
                                        len(graph_erer.triplesA)%batchSize)])
        self.triplesB = np.array(graph_erer.triplesB[0: (len(graph_erer.triplesB) -
                                        len(graph_erer.triplesB)%batchSize)])
        logger.info("Shape of triples A: %s", str(self.triplesA.shape))
        logger.info("Shape of triples B: %s", str(self.triplesB.shape))

        #Collect Negative Samples
        self.ntriplesA = np.array(get_negative_samples(graph_erer.triplesA, len(self.entityA),
                            len(self.entityA), len(self.relationA), graph_erer.entity_pairs,
                            neg_rate=neg_rate, neg_rel_rate=neg_rel_rate))
        self.ntriplesB = np.array(get_negative_samples(graph_erer.triplesB, len(self.entityB),
                            len(self.entityB), len(self.relationB), graph_erer.entity_pairs,
                            neg_rate=neg_rate, neg_rel_rate=neg_rel_rate))
        logger.info("Shape of negative triples A: %s", str(self.ntriplesA.shape))
        logger.info("Shape of negative triples B: %s", str(self.ntriplesB.shape))

        # List of ILLs / linked entities. Remove last incomplete batch if any.
        logger.info("Shape of prior_pairs in GRAPH: %d", len(graph_erer.prior_pairs))

        self.prior_pairs = np.array(graph_erer.prior_pairs[0: (len(graph_erer.prior_pairs) -
                                        len(graph_erer.prior_pairs)%batchSize)])
        dummy_rel = -1 #using just to reuse neg sampling method for triplets.
        pp_triples = [(a,b, dummy_rel) for a,b in self.prior_pairs]
        neg_pp_triples= get_negative_samples(pp_triples, len(self.entityA),len(self.entityB),
                                1, graph_erer.entity_pairs, neg_rate=neg_rate, neg_rel_rate=0)
        #Note: neg_rel_rate is 0 because we don't want to relace r but only h or t.

        self.neg_prior_pairs = np.array([(h, t) for (h, t, r) in neg_pp_triples])
        logger.info("Shape of prior_pairs: %s", str(self.prior_pairs.shape))
        logger.info("Shape of negative prior_pairs: %s", str(self.neg_prior_pairs.shape))

        #Define Embedding Variables
        initializer = tf.contrib.layers.xavier_initializer(uniform = True)
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
        self.headA = tf.placeholder(tf.int32, shape=[self.batchSize, 1])
        self.tailA = tf.placeholder(tf.int32, shape=[self.batchSize, 1])
        self.relA = tf.placeholder(tf.int32, shape=[self.batchSize, 1])
        self.neg_headA = tf.placeholder(tf.int32, shape=[self.batchSize, (neg_rate + neg_rel_rate)])
        self.neg_tailA = tf.placeholder(tf.int32, shape=[self.batchSize, (neg_rate + neg_rel_rate)])
        self.neg_relA = tf.placeholder(tf.int32, shape=[self.batchSize, (neg_rate + neg_rel_rate)])

        self.headB = tf.placeholder(tf.int32, shape=[self.batchSize, 1])
        self.tailB = tf.placeholder(tf.int32, shape=[self.batchSize, 1])
        self.relB = tf.placeholder(tf.int32, shape=[self.batchSize, 1])
        self.neg_headB = tf.placeholder(tf.int32, shape=[self.batchSize, (neg_rate + neg_rel_rate)])
        self.neg_tailB = tf.placeholder(tf.int32, shape=[self.batchSize, (neg_rate + neg_rel_rate)])
        self.neg_relB = tf.placeholder(tf.int32, shape=[self.batchSize, (neg_rate + neg_rel_rate)])

        self.ent_A = tf.placeholder(tf.int32, shape=[self.batchSize, 1])
        self.ent_B = tf.placeholder(tf.int32, shape=[self.batchSize, 1])
        self.neg_ent_A = tf.placeholder(tf.int32, shape=[self.batchSize, neg_rate])
        self.neg_ent_B = tf.placeholder(tf.int32, shape=[self.batchSize, neg_rate])

        pos_ha = tf.nn.embedding_lookup(self.ent_embeddings_A, self.headA)
        pos_ta = tf.nn.embedding_lookup(self.ent_embeddings_A, self.tailA)
        pos_ra = tf.nn.embedding_lookup(self.rel_embeddings_A, self.relA)
        pos_nha = tf.nn.embedding_lookup(self.ent_embeddings_A, self.neg_headA)
        pos_nta = tf.nn.embedding_lookup(self.ent_embeddings_A, self.neg_tailA)
        pos_nra = tf.nn.embedding_lookup(self.rel_embeddings_A, self.neg_relA)

        pos_hb = tf.nn.embedding_lookup(self.ent_embeddings_B, self.headB)
        pos_tb = tf.nn.embedding_lookup(self.ent_embeddings_B, self.tailB)
        pos_rb = tf.nn.embedding_lookup(self.rel_embeddings_B, self.relB)
        pos_nhb = tf.nn.embedding_lookup(self.ent_embeddings_B, self.neg_headB)
        pos_ntb = tf.nn.embedding_lookup(self.ent_embeddings_B, self.neg_tailB)
        pos_nrb = tf.nn.embedding_lookup(self.rel_embeddings_B, self.neg_relB)

        pos_entA = tf.nn.embedding_lookup(self.ent_embeddings_A, self.ent_A)
        pos_entB = tf.nn.embedding_lookup(self.ent_embeddings_B, self.ent_B)
        pos_nentA = tf.nn.embedding_lookup(self.ent_embeddings_A, self.neg_ent_A)
        pos_nentB = tf.nn.embedding_lookup(self.ent_embeddings_B, self.neg_ent_B)
        pos_ent_trans = tf.nn.embedding_lookup(self.ent_translation, range(0, dimension))

        #Normalize embeddings
        pos_ha = tf.nn.l2_normalize(pos_ha, [1,2])
        pos_ta = tf.nn.l2_normalize(pos_ta, [1,2])
        pos_ra = tf.nn.l2_normalize(pos_ra, [1,2])
        pos_nha = tf.nn.l2_normalize(pos_nha, [1,2])
        pos_nta = tf.nn.l2_normalize(pos_nta, [1,2])
        pos_nra = tf.nn.l2_normalize(pos_nra, [1,2])

        pos_hb = tf.nn.l2_normalize(pos_hb, [1,2])
        pos_tb = tf.nn.l2_normalize(pos_tb, [1,2])
        pos_rb = tf.nn.l2_normalize(pos_rb, [1,2])
        pos_nhb = tf.nn.l2_normalize(pos_nhb, [1,2])
        pos_ntb = tf.nn.l2_normalize(pos_ntb, [1,2])
        pos_nrb = tf.nn.l2_normalize(pos_nrb, [1,2])

        pos_entA = tf.nn.l2_normalize(pos_entA, [1,2])
        pos_entB = tf.nn.l2_normalize(pos_entB, [1,2])
        pos_nentA = tf.nn.l2_normalize(pos_nentA, [1,2])
        pos_nentB = tf.nn.l2_normalize(pos_nentB, [1,2])
        pos_ent_trans = tf.nn.l2_normalize(pos_ent_trans, 1)

        logger.info("Triple Shapes A: %s, %s, %s", str(pos_ha.shape), str(pos_ta.shape), str(pos_ra.shape))
        logger.info("Triple Shapes B: %s, %s, %s", str(pos_hb.shape), str(pos_tb.shape), str(pos_rb.shape))
        logger.info("Prior Pairs Shape: %s, %s, %s", str(pos_entA.shape), str(pos_entB.shape), str(pos_ent_trans.shape))

        #Compute loss: knowledge model
        _ap_score = self._calc(pos_ha, pos_ta, pos_ra)
        _an_score = self._calc(pos_nha, pos_nta, pos_nra)
        #Note: can use also: tf.abs(tf.subtract(tf.add(pos_ha, pos_ra), post_ta))

        ap_score = tf.reduce_sum(tf.reduce_mean(_ap_score, 1, keepdims = False), 1, keepdims = True)
        an_score = tf.reduce_sum(tf.reduce_mean(_an_score, 1, keepdims = False), 1, keepdims = True)

        _bp_score = self._calc(pos_hb, pos_tb, pos_rb)
        _bn_score = self._calc(pos_nhb, pos_ntb, pos_nrb)

        bp_score =  tf.reduce_sum(tf.reduce_mean(_bp_score, 1, keepdims = False), 1, keepdims = True)
        bn_score =  tf.reduce_sum(tf.reduce_mean(_bn_score, 1, keepdims = False), 1, keepdims = True)

        self.a_score = tf.reduce_sum(tf.maximum(ap_score - an_score + self.margin, 0))
        self.b_score = tf.reduce_sum(tf.maximum(bp_score - bn_score + self.margin, 0))
        logger.info("AScore Shape %s. Bscore Shape: %s", str(self.a_score.shape), str(self.b_score.shape))


        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)#AdamOptimizer(self.learning_rate)
        self.optimizer_A = self.optimizer.minimize(self.a_score,
                                var_list=[self.ent_embeddings_A, self.rel_embeddings_A])
        self.optimizer_B = self.optimizer.minimize(self.b_score,
                                var_list=[self.ent_embeddings_B, self.rel_embeddings_B])

        #Compute loss: alignment model
        _p_alignment_loss = self._calc_alignment(tf.reshape(pos_entA, [self.batchSize, self.dimension]),
                                                tf.reshape(pos_entB, [self.batchSize, self.dimension]),
                                                pos_ent_trans)
        _n_alignment_loss = self._calc_alignment(tf.reshape(pos_nentA, [self.batchSize * neg_rate, self.dimension]),
                                                tf.reshape(pos_nentB, [self.batchSize * neg_rate, self.dimension]),
                                                pos_ent_trans)
        #Note: can also use: tf.abs(tf.subtract(tf.matmul(pos_entA, pos_ent_trans), pos_entB))

        _p_alignment_loss = tf.reshape(_p_alignment_loss, [self.batchSize, 1, self.dimension])
        _n_alignment_loss = tf.reshape(_n_alignment_loss, [self.batchSize, neg_rate, self.dimension])

        logger.info("Shape of P Align Score: %s", str(_p_alignment_loss.shape))
        logger.info("Shape of N Align Score: %s", str(_n_alignment_loss.shape))
        p_alignment_loss =  tf.reduce_sum(tf.reduce_mean(_p_alignment_loss, 1, keepdims = False), 1, keepdims = True)
        n_alignment_loss =  tf.reduce_sum(tf.reduce_mean(_n_alignment_loss, 1, keepdims = False), 1, keepdims = True)

        self.alignment_loss = tf.reduce_sum(tf.maximum(p_alignment_loss - n_alignment_loss + self.margin, 0))
        logger.info("Alignment loss Shape %s.", str(self.alignment_loss.shape))

        #Configure optimizer
        self.optimizer_AM = self.optimizer.minimize(self.alignment_loss * self.alpha,
                                    var_list=[self.ent_translation, self.ent_embeddings_A,
                                            self.ent_embeddings_B])

        #Configure session
        self.sess = tf.Session()

        #Collect summary for tensorboard
        tf.summary.scalar('p_alignment_loss', tf.reduce_mean(p_alignment_loss), collections=['align'])
        tf.summary.scalar('n_alignment_loss', tf.reduce_mean(n_alignment_loss), collections=['align'])
        tf.summary.scalar('alignment_loss', self.alignment_loss, collections=['align'])
        tf.summary.scalar('bp_score', tf.reduce_mean(bp_score), collections=['b'])
        tf.summary.scalar('bn_score', tf.reduce_mean(bn_score), collections=['b'])
        tf.summary.scalar('b_loss', self.b_score, collections=['b'])
        tf.summary.scalar('ap_score', tf.reduce_mean(ap_score), collections=['a'])
        tf.summary.scalar('an_score', tf.reduce_mean(an_score), collections=['a'])
        tf.summary.scalar('a_loss', self.a_score, collections=['a'])

        self.merged_align = tf.summary.merge_all(key='align')
        self.merged_a = tf.summary.merge_all(key='a')
        self.merged_b = tf.summary.merge_all(key='b')
        self.align_summary_writer = tf.summary.FileWriter(get_tf_summary_file_path(logger) + '_align', self.sess.graph)
        self.a_summary_writer = tf.summary.FileWriter(get_tf_summary_file_path(logger) +'_a', self.sess.graph)
        self.b_summary_writer = tf.summary.FileWriter(get_tf_summary_file_path(logger) +'_b', self.sess.graph)

    def _calc(self, h, t, r):
        """
            TransE objective function.
            It estimates embeddings as translation from head to tail entity.
        """
        return abs(h + r - t)

    def _calc_alignment(self, a, b, et):
        """
            Alignment model objectice function.
            It first projects entity to embedding of other KG and then compute distance.
        """
        return abs(tf.matmul(a, et) - b)


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
                    neg_batchend = min(len(self.ntriplesA), i + self.batchSize * (self.neg_rel_rate + self.neg_rate))
                    feed_dict = {
                        self.headA : self.triplesA[i:batchend][:, 0].reshape(self.batchSize, 1),
                        self.tailA : self.triplesA[i:batchend][:, 1].reshape(self.batchSize, 1),
                        self.relA  : self.triplesA[i:batchend][:, 2].reshape(self.batchSize, 1),
                        self.neg_headA : self.ntriplesA[i:neg_batchend][:, 0].reshape(self.batchSize, (self.neg_rel_rate + self.neg_rate)),
                        self.neg_tailA : self.ntriplesA[i:neg_batchend][:, 1].reshape(self.batchSize, (self.neg_rel_rate + self.neg_rate)),
                        self.neg_relA  : self.ntriplesA[i:neg_batchend][:, 2].reshape(self.batchSize, (self.neg_rel_rate + self.neg_rate)),
                        }
                    if batchend == len(self.triplesA):
                        _ , cur_loss, summary = self.sess.run([self.optimizer_A, self.a_score, self.merged_a],
                                                        feed_dict=feed_dict)
                        self.a_summary_writer.add_summary(summary, epoch)
                    else:
                        _ , cur_loss = self.sess.run([self.optimizer_A, self.a_score], feed_dict=feed_dict)
                    a_loss = a_loss + cur_loss

                #Learning embeddings for Knowledge Graph B
                b_loss = 0
                for i in np.arange(0, len(self.triplesB), self.batchSize):
                    batchend = min(len(self.triplesB), i + self.batchSize)
                    neg_batchend = min(len(self.ntriplesB), i + self.batchSize * (self.neg_rel_rate + self.neg_rate))

                    feed_dict = {
                        self.headB : self.triplesB[i:batchend][:, 0].reshape(self.batchSize, 1),
                        self.tailB : self.triplesB[i:batchend][:, 1].reshape(self.batchSize, 1),
                        self.relB  : self.triplesB[i:batchend][:, 2].reshape(self.batchSize, 1),
                        self.neg_headB : self.ntriplesB[i:neg_batchend][:, 0].reshape(self.batchSize, (self.neg_rel_rate + self.neg_rate)),
                        self.neg_tailB : self.ntriplesB[i:neg_batchend][:, 1].reshape(self.batchSize, (self.neg_rel_rate + self.neg_rate)),
                        self.neg_relB  : self.ntriplesB[i:neg_batchend][:, 2].reshape(self.batchSize, (self.neg_rel_rate + self.neg_rate)),
                        }
                    if batchend == len(self.triplesB):
                        _ , cur_loss, summary = self.sess.run([self.optimizer_B, self.b_score, self.merged_b],
                                                    feed_dict=feed_dict)
                        self.b_summary_writer.add_summary(summary, epoch)
                    else:
                        _ , cur_loss = self.sess.run([self.optimizer_B, self.b_score], feed_dict=feed_dict)
                    b_loss = b_loss + cur_loss

                #Learning alignment model
                am_loss = 0
                for i in np.arange(0, len(self.prior_pairs), self.batchSize):
                    batchend = min(len(self.prior_pairs), i + self.batchSize)
                    neg_batchend = min(len(self.neg_prior_pairs), i + self.batchSize *  self.neg_rate)
                    feed_dict = {
                        self.ent_A : self.prior_pairs[i:batchend][:, 0].reshape(self.batchSize, 1),
                        self.ent_B : self.prior_pairs[i:batchend][:, 1].reshape(self.batchSize, 1),
                        self.neg_ent_A : self.neg_prior_pairs[i:neg_batchend][:, 0].reshape(self.batchSize, self.neg_rate),
                        self.neg_ent_B : self.neg_prior_pairs[i:neg_batchend][:, 1].reshape(self.batchSize, self.neg_rate),
                        }
                    if batchend == len(self.prior_pairs):
                        _ , cur_loss, summary = self.sess.run([self.optimizer_AM, self.alignment_loss, self.merged_align],
                                                        feed_dict=feed_dict)
                        self.align_summary_writer.add_summary(summary, epoch)
                    else:
                        _ , cur_loss = self.sess.run([self.optimizer_AM, self.alignment_loss], feed_dict=feed_dict)
                    am_loss = am_loss + cur_loss

                loss = a_loss + b_loss + am_loss
                if loss:
                    logger.info("Epoch: %d a_loss:%f b_loss:%f  am_loss:%f Loss: %f",
                        epoch, a_loss, b_loss, am_loss, loss)
                else:
                    logger.info("Zero Loss, finish training in %d epochs", epoch)
                    break
        return loss

    def get_ent_embeddings_A(self):
        with self.sess.as_default():
            ent_A = tf.nn.embedding_lookup(self.ent_embeddings_A, range(0, len(self.entityA))).eval()
            ent_trans = tf.nn.embedding_lookup(self.ent_translation, range(0, self.dimension)).eval()
            return tf.matmul(ent_A, ent_trans).eval()

    def get_ent_embeddings_B(self):
        with self.sess.as_default():
            return tf.nn.embedding_lookup(self.ent_embeddings_B, range(0, len(self.entityB))).eval()

    def close_tf_session(self):
        tf.reset_default_graph()
        self.sess.close()

    def __str__(self):
        return "MTransE"

    def __del__(self):
        self.close_tf_session()
