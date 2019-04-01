import tensorflow as tf
import numpy as np

from common import get_logger, get_negative_samples, get_tf_summary_file_path
from scipy import spatial
import timeit

logger = get_logger("RL.ERER.ETransE.")

class ETransE(object):
    """
        Evolution TransE. Models two KGs separtely in different vector space.
        Uses a projection matrix (M) to project entity from one vector space to the other.
        Uses evolution vector (E_r) to translate entities considering growth during two census.
        e.g. (A, solter, "civil") in KG1 (1930) and (B, casat, "civil") in KG2 (1940)
        Positive_Loss_A = | A + "civil" - solter|
        Positive_Loss_B = |B + "civil" - casat|
        Positive_Loss_Proj = |A.M - B|
        Positive_Loss_Evol = |solter.M + E_r - casat|
    """

    def __init__(self, graph_erer, dimension=64, batchSize=128, learning_rate=0.1, alpha=5,
                        margin=1, neg_rate=7, neg_rel_rate=1, regularizer_scale=0.1, beta=5):
        self.entityA = graph_erer.entityA
        self.entityB = graph_erer.entityB
        self.relationA = graph_erer.relationA
        self.relationB = graph_erer.relationB

        self.dimension = dimension #Embedding Dimension
        self.batchSize = batchSize #BatchSize for Stochastic Gradient Decent
        self.learning_rate = learning_rate #Learning rate for optmizer
        self.neg_rel_rate = neg_rel_rate
        self.neg_rate = neg_rate
        self.margin = margin
        self.alpha = alpha
        self.beta = beta

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
        self.prior_pairs = np.array(graph_erer.prior_pairs[0: (len(graph_erer.prior_pairs) -
                                        len(graph_erer.prior_pairs)%batchSize)])
        dummy_rel = -1 #using just to reuse neg sampling method for triplets.
        pp_triples = [(a, b, dummy_rel) for a,b in self.prior_pairs]
        neg_pp_triples= np.array(get_negative_samples(pp_triples, len(self.entityA),len(self.entityB),
                                1, graph_erer.entity_pairs, neg_rate=neg_rate, neg_rel_rate=0))
        #Note: neg_rel_rate is 0 because we don't want to relace r but only h or t.

        self.neg_prior_pairs = np.array([(h, t) for (h, t, r) in neg_pp_triples])
        logger.info("Shape of prior_pairs: %s", str(self.prior_pairs.shape))
        logger.info("Shape of negative prior_pairs: %s", str(self.neg_prior_pairs.shape))

        self.unique_rels = list(set(self.relationA + self.relationB))
        logger.info("No. of unique relations: %s", len(self.unique_rels))
        self.relA_to_urel_map = {}
        for i in range(len(graph_erer.relationA)):
            self.relA_to_urel_map[i] = self.unique_rels.index(graph_erer.relationA[i])
        self.relB_to_urel_map = {}
        for i in range(len(graph_erer.relationB)):
            self.relB_to_urel_map[i] = self.unique_rels.index(graph_erer.relationB[i])

        #Generate Evolution Pairs from Prior Pairs
        self.evolution_pairs = []
        for (a, b) in self.prior_pairs:
            a_triples = [(h, t, r) for (h, t, r) in self.triplesA if h == a]
            b_triples = [(h, t, r) for (h, t, r) in self.triplesB if h == b]
            for (ah, at, ar) in a_triples:
                unique_rel_indexA = self.relA_to_urel_map[int(ar)]
                bt = [t for (h, t, r) in b_triples if unique_rel_indexA == \
                                self.relB_to_urel_map[int(r)]]
                if len(bt):
                    self.evolution_pairs.append((at, bt[0], unique_rel_indexA))

        self.evolution_pairs = np.array(self.evolution_pairs[0: (len(self.evolution_pairs) -
                                        len(self.evolution_pairs)%self.batchSize)])
        self.neg_evolution_pairs= np.array(get_negative_samples(self.evolution_pairs, len(self.entityA),
                                len(self.entityB), 1, graph_erer.entity_pairs,
                                neg_rate=neg_rate, neg_rel_rate=0))
        #Note: neg_rel_rate is 0 because we don't want to relace r but only h or t.

        logger.info("No. of evolution_pairs: %s", str(self.evolution_pairs.shape))
        logger.info("Shape of negative evolution_pairs: %s", str(self.neg_evolution_pairs.shape))
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
        self.projection_matrix = tf.get_variable(name = "projection_matrix", shape = [dimension, dimension],
                                    initializer = initializer, regularizer = regularizer)
        self.evolution_vectors = tf.get_variable(name = "evolution_vectors", shape = [len(self.unique_rels), dimension],
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

        self.evolve_A = tf.placeholder(tf.int32, shape=[self.batchSize, 1])
        self.evolve_B = tf.placeholder(tf.int32, shape=[self.batchSize, 1])
        self.evolve_rel = tf.placeholder(tf.int32, shape=[self.batchSize, 1])
        self.neg_evolve_A = tf.placeholder(tf.int32, shape=[self.batchSize, neg_rate])
        self.neg_evolve_B = tf.placeholder(tf.int32, shape=[self.batchSize, neg_rate])
        self.neg_evolve_rel = tf.placeholder(tf.int32, shape=[self.batchSize, neg_rate])


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
        pos_ent_proj = tf.nn.embedding_lookup(self.projection_matrix, range(0, dimension))

        pos_evolA = tf.nn.embedding_lookup(self.ent_embeddings_A, self.evolve_A)
        pos_evolB = tf.nn.embedding_lookup(self.ent_embeddings_B, self.evolve_B)
        pos_evolve_vec = tf.nn.embedding_lookup(self.evolution_vectors, self.evolve_rel)
        pos_nevolA = tf.nn.embedding_lookup(self.ent_embeddings_A, self.neg_evolve_A)
        pos_nevolB = tf.nn.embedding_lookup(self.ent_embeddings_B, self.neg_evolve_B)
        pos_nevolve_vec = tf.nn.embedding_lookup(self.evolution_vectors, self.neg_evolve_rel)

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
        pos_ent_proj = tf.nn.l2_normalize(pos_ent_proj, 1)

        pos_evolA = tf.nn.l2_normalize(pos_evolA, [1,2])
        pos_evolB = tf.nn.l2_normalize(pos_evolB, [1,2])
        pos_evolve_vec = tf.nn.l2_normalize(pos_evolve_vec, 1)
        pos_nevolA = tf.nn.l2_normalize(pos_nevolA, [1,2])
        pos_nevolB = tf.nn.l2_normalize(pos_nevolB, [1,2])
        pos_nevolve_vec = tf.nn.l2_normalize(pos_nevolve_vec, 1)

        logger.info("Triple Shapes A: %s, %s, %s", str(pos_ha.shape), str(pos_ta.shape), str(pos_ra.shape))
        logger.info("Triple Shapes B: %s, %s, %s", str(pos_hb.shape), str(pos_tb.shape), str(pos_rb.shape))
        logger.info("Prior Pairs Shape: %s, %s, %s", str(pos_entA.shape), str(pos_entB.shape), str(pos_ent_proj.shape))
        logger.info("Evolution Pairs Shape: %s, %s, %s", str(pos_evolA.shape), str(pos_evolB.shape), str(pos_evolve_vec.shape))

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
                                                pos_ent_proj)
        _n_alignment_loss = self._calc_alignment(tf.reshape(pos_nentA, [self.batchSize * neg_rate, self.dimension]),
                                                tf.reshape(pos_nentB, [self.batchSize * neg_rate, self.dimension]),
                                                pos_ent_proj)
        #Note: can also use: tf.abs(tf.subtract(tf.matmul(pos_entA, pos_ent_proj), pos_entB))

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
                                    var_list=[self.projection_matrix, self.ent_embeddings_A,
                                            self.ent_embeddings_B])

        logger.info("Shape of PosA Evolve: %s", str(pos_evolA.shape))
        logger.info("Shape of PosB Evolve: %s", str(pos_evolB.shape))
        logger.info("Shape of project Evolve: %s", str(pos_ent_proj.shape))
        logger.info("Shape of Evolve Vector: %s", str(pos_evolve_vec.shape))
        #Compute Evolution Loss
        _p_evolve_score = self._calc_evolve(tf.reshape(pos_evolA, [self.batchSize, self.dimension]),
                                        tf.reshape(pos_evolB, [self.batchSize, self.dimension]),
                                        pos_ent_proj,
                                        tf.reshape(pos_evolve_vec, [self.batchSize, self.dimension]))

        _n_evolve_score = self._calc_evolve(tf.reshape(pos_nevolA, [self.batchSize * neg_rate, self.dimension]),
                                        tf.reshape(pos_nevolB, [self.batchSize * neg_rate, self.dimension]),
                                        pos_ent_proj,
                                        tf.reshape(pos_nevolve_vec, [self.batchSize * neg_rate, self.dimension]))

        _p_evolve_score = tf.reshape(_p_evolve_score, [self.batchSize, 1, self.dimension])
        _n_evolve_score = tf.reshape(_n_evolve_score, [self.batchSize, neg_rate, self.dimension])

        logger.info("Shape of P Evolve Score: %s", str(_p_evolve_score.shape))
        logger.info("Shape of N Evolve Score: %s", str(_n_evolve_score.shape))
        p_evolve_score =  tf.reduce_sum(tf.reduce_mean(_p_evolve_score, 1, keepdims = False), 1, keepdims = True)
        n_evolve_score =  tf.reduce_sum(tf.reduce_mean(_n_evolve_score, 1, keepdims = False), 1, keepdims = True)

        self.evolution_loss = tf.reduce_sum(tf.maximum(p_evolve_score - n_evolve_score + self.margin, 0))
        logger.info("Evolution loss Shape %s.", str(self.evolution_loss.shape))

        #Configure optimizer
        self.optimizer_evolve = self.optimizer.minimize(self.evolution_loss * self.beta,
                                    var_list=[self.projection_matrix, self.ent_embeddings_A,
                                            self.ent_embeddings_B, self.evolution_vectors])

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
        tf.summary.scalar('n_evolve_score', tf.reduce_mean(n_evolve_score), collections=['evolve'])
        tf.summary.scalar('p_evolve_score', tf.reduce_mean(p_evolve_score), collections=['evolve'])
        tf.summary.scalar('evolution_loss', self.evolution_loss, collections=['evolve'])

        self.merged_align = tf.summary.merge_all(key='align')
        self.merged_a = tf.summary.merge_all(key='a')
        self.merged_b = tf.summary.merge_all(key='b')
        self.merged_evolve = tf.summary.merge_all(key='evolve')
        self.align_summary_writer = tf.summary.FileWriter(get_tf_summary_file_path(logger) + '_align', self.sess.graph)
        self.a_summary_writer = tf.summary.FileWriter(get_tf_summary_file_path(logger) +'_a', self.sess.graph)
        self.b_summary_writer = tf.summary.FileWriter(get_tf_summary_file_path(logger) +'_b', self.sess.graph)
        self.evolve_summary_writer = tf.summary.FileWriter(get_tf_summary_file_path(logger) +'_evolve', self.sess.graph)

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

    def _calc_evolve(self, a, b, et, evolve_vec):
        """
            Evolution Model objective function.
            Project Entity to other vector space and add evolution vector.
        """
        return abs(tf.matmul(a, et) + evolve_vec - b)

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

                #Learning evolution model
                evolve_loss = 0
                for i in np.arange(0, len(self.evolution_pairs), self.batchSize):
                    batchend = min(len(self.evolution_pairs), i + self.batchSize)
                    neg_batchend = min(len(self.neg_evolution_pairs), i + self.batchSize *  self.neg_rate)
                    feed_dict = {
                        self.evolve_A : self.evolution_pairs[i:batchend][:, 0].reshape(self.batchSize, 1),
                        self.evolve_B : self.evolution_pairs[i:batchend][:, 1].reshape(self.batchSize, 1),
                        self.evolve_rel : self.evolution_pairs[i:batchend][:, 2].reshape(self.batchSize, 1),
                        self.neg_evolve_A : self.neg_evolution_pairs[i:neg_batchend][:, 0].reshape(self.batchSize, self.neg_rate),
                        self.neg_evolve_B : self.neg_evolution_pairs[i:neg_batchend][:, 1].reshape(self.batchSize, self.neg_rate),
                        self.neg_evolve_rel : self.neg_evolution_pairs[i:neg_batchend][:, 2].reshape(self.batchSize, self.neg_rate),
                        }
                    if batchend == len(self.evolution_pairs):
                        _ , cur_loss, summary = self.sess.run([self.optimizer_evolve, self.evolution_loss, self.merged_evolve],
                                                        feed_dict=feed_dict)
                        self.evolve_summary_writer.add_summary(summary, epoch)
                    else:
                        _ , cur_loss = self.sess.run([self.optimizer_evolve, self.evolution_loss], feed_dict=feed_dict)
                    evolve_loss = evolve_loss + cur_loss

                loss = a_loss + b_loss + am_loss + evolve_loss
                if loss:
                    logger.info("Epoch: %d a_loss:%f b_loss:%f  am_loss:%f evolve_loss:%f Loss: %f",
                        epoch, a_loss, b_loss, am_loss, evolve_loss, loss)
                else:
                    logger.info("Zero Loss, finish training in %d epochs", epoch)
                    break
        return loss

    def get_ent_embeddings_A(self):
        with self.sess.as_default():
            ent_A = tf.nn.embedding_lookup(self.ent_embeddings_A, range(0, len(self.entityA))).eval()
            ent_proj = tf.nn.embedding_lookup(self.projection_matrix, range(0, self.dimension)).eval()
            return tf.matmul(ent_A, ent_proj).eval()

    def get_ent_embeddings_B(self):
        with self.sess.as_default():
            return tf.nn.embedding_lookup(self.ent_embeddings_B, range(0, len(self.entityB))).eval()

    def close_tf_session(self):
        tf.reset_default_graph()
        self.sess.close()

    def __str__(self):
        return "ETransE"

    def __del__(self):
        self.close_tf_session()




