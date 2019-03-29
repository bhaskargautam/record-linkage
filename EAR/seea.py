import numpy as np
import tensorflow as tf

from common import get_logger, get_negative_samples, get_tf_summary_file_path
from scipy import spatial

logger = get_logger('RL.EAR.SEEA')

class SEEA(object):

    #Special Weights for few attributes (attr, weight)
    #Attr energy function is multiplied by weight
    special_attr_weight_dict = {
        'title' : 8.0, #For Cora
        'surname' : 8.0, #For FEBRL
        'yob' : 8.0 #For Census
    }

    def __init__(self, graph_ear, dimension=10, learning_rate=0.1, batchSize=100,
                        margin=1, regularizer_scale = 0.1, neg_rate=1, neg_rel_rate=0):
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
        self.entity_pairs = graph_ear.entity_pairs
        self.true_pairs = graph_ear.true_pairs

        #Build attr weight vector
        self.attr_weights = []
        for a in self.attribute:
            self.attr_weights.append(SEEA.special_attr_weight_dict.get(a, 1.0))
        logger.info("Using attr weights as: %s", str(self.attr_weights))
        self.attr_weights = tf.constant(self.attr_weights, dtype=tf.float32)
        logger.info("Attr wt Tensor: %s", str(self.attr_weights))

        # List of triples. Remove last incomplete batch if any.
        self.atriples = np.array(graph_ear.atriples[0: (len(graph_ear.atriples) -
                                            len(graph_ear.atriples)%batchSize)])
        self.rtriples = np.array(graph_ear.rtriples[0: (len(graph_ear.rtriples) -
                                            len(graph_ear.rtriples)%batchSize)])
        logger.info("Modified Atriples size: %d", len(self.atriples))
        logger.info("Modified Rtriples size: %d", len(self.rtriples))

        #Collect Negative Samples
        self.natriples = np.array(get_negative_samples(self.atriples, len(self.entity),
                                        len(self.value), len(self.attribute), [],
                                        neg_rate=neg_rate, neg_rel_rate =neg_rel_rate))
        self.nrtriples = np.array(get_negative_samples(self.rtriples, len(self.entity),
                                        len(self.entity), len(self.relation), self.entity_pairs,
                                        neg_rate=neg_rate, neg_rel_rate=neg_rel_rate))

        #Define Embedding Variables
        initializer = tf.contrib.layers.xavier_initializer(uniform = False)
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

        #Load Attr Weights
        self.p_attr_wt = tf.map_fn(lambda a: tf.map_fn(lambda x: self.attr_weights[x], a, dtype=tf.float32), self.attr, dtype=tf.float32)
        self.n_attr_wt = tf.map_fn(lambda a: tf.map_fn(lambda x: self.attr_weights[x], a, dtype=tf.float32), self.neg_attr, dtype=tf.float32)
        logger.info("Tensor of Pos Attr Wt.: %s", str(self.p_attr_wt))
        logger.info("Tensor of Neg Attr Wt.: %s", str(self.n_attr_wt))
        self.p_attr_wt = tf.cast(tf.tile(tf.expand_dims(self.p_attr_wt, 2), [1, 1, self.dimension]), tf.float32)
        self.n_attr_wt = tf.cast(tf.tile(tf.expand_dims(self.n_attr_wt, 2), [1, 1 ,self.dimension]), tf.float32)
        logger.info("Tensor of Pos Attr Wt.: %s", str(self.p_attr_wt))
        logger.info("Tensor of Neg Attr Wt.: %s", str(self.n_attr_wt))

        #Load Embedding Vectors
        pos_h = tf.nn.embedding_lookup(self.ent_embeddings, self.head)
        pos_t = tf.nn.embedding_lookup(self.ent_embeddings, self.tail)
        pos_r = tf.nn.embedding_lookup(self.rel_embeddings, self.rel)
        pos_nh = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_head)
        pos_nt = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_tail)
        pos_nr = tf.nn.embedding_lookup(self.rel_embeddings, self.neg_rel)
        pos_attr_h = tf.nn.embedding_lookup(self.ent_embeddings, self.attr_head)
        pos_val = tf.nn.embedding_lookup(self.val_embeddings, self.val)
        pos_attr = tf.nn.embedding_lookup(self.attr_embeddings, self.attr)
        pos_attr_nh = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_attr_head)
        pos_attr_nv = tf.nn.embedding_lookup(self.val_embeddings, self.neg_val)
        pos_attr_na = tf.nn.embedding_lookup(self.attr_embeddings, self.neg_attr)
        pos_proj = tf.nn.embedding_lookup(self.projection_matrix, self.attr)
        pos_nproj = tf.nn.embedding_lookup(self.projection_matrix, self.neg_attr)

        #Normalize Vectors
        pos_h = tf.nn.l2_normalize(pos_h, [1,2])
        pos_t = tf.nn.l2_normalize(pos_t, [1,2])
        pos_r = tf.nn.l2_normalize(pos_r, [1,2])
        pos_nh = tf.nn.l2_normalize(pos_nh, [1,2])
        pos_nt = tf.nn.l2_normalize(pos_nt, [1,2])
        pos_nr = tf.nn.l2_normalize(pos_nr, [1,2])
        pos_attr_h = tf.nn.l2_normalize(pos_attr_h, [1,2])
        pos_val = tf.nn.l2_normalize(pos_val, [1,2])
        pos_attr = tf.nn.l2_normalize(pos_attr, [1,2])
        pos_attr_nh = tf.nn.l2_normalize(pos_attr_nh, [1,2])
        pos_attr_nv = tf.nn.l2_normalize(pos_attr_nv, [1,2])
        pos_attr_na = tf.nn.l2_normalize(pos_attr_na, [1,2])
        pos_proj = tf.nn.l2_normalize(pos_proj, [1,2])
        pos_nproj = tf.nn.l2_normalize(pos_nproj, [1,2])

        #Project Entities to attribute space
        proj_pos_attr_h = self._transfer(pos_attr_h, pos_proj)
        proj_pos_attr_nh = self._transfer(pos_attr_nh, pos_nproj)

        #Compute Loss
        _p_score = self._calc(pos_h, pos_t, pos_r)
        _n_score = self._calc(pos_nh, pos_nt, pos_nr)

        _ap_score = self._attr_calc(proj_pos_attr_h, pos_val, pos_attr)
        _an_score = self._attr_calc(proj_pos_attr_nh, pos_attr_nv, pos_attr_na)
        logger.info("Shape of APSCORE.: %s", str(_ap_score.shape))
        logger.info("Shape of ANSCORE.: %s", str(_an_score.shape))

        _wap_score = tf.math.multiply(_ap_score, self.p_attr_wt)
        _wan_score = tf.math.multiply(_an_score, self.n_attr_wt)
        logger.info("Shape of APSCORE.: %s", str(_wap_score.shape))
        logger.info("Shape of ANSCORE.: %s", str(_wan_score.shape))

        p_score = tf.reduce_sum(tf.reduce_mean(_p_score, 1, keepdims=False), axis=1, keepdims=True)
        n_score = tf.reduce_sum(tf.reduce_mean(_n_score, 1, keepdims=False), axis=1, keepdims=True)
        ap_score = tf.reduce_sum(tf.reduce_mean(_wap_score, 1, keepdims=False), axis=1, keepdims=True)
        an_score = tf.reduce_sum(tf.reduce_mean(_wan_score, 1, keepdims=False), axis=1, keepdims=True)
        logger.info("Shape of APSCORE*.: %s", str(ap_score.shape))
        logger.info("Shape of ANSCORE*.: %s", str(an_score.shape))
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
        return e - tf.reduce_sum(e * n, 1, keepdims = True) * n

    def train(self, max_epochs=100, iter_num=0):
        loss = 0
        with self.sess.as_default():
            #self.sess.run(tf.global_variables_initializer())
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
                        self.neg_head : self.nrtriples[i:neg_batchend][:,0].reshape(self.batchSize, self.neg_rel_rate + self.neg_rate),
                        self.neg_tail : self.nrtriples[i:neg_batchend][:,1].reshape(self.batchSize, self.neg_rel_rate + self.neg_rate),
                        self.neg_rel : self.nrtriples[i:neg_batchend][:,2].reshape(self.batchSize, self.neg_rel_rate + self.neg_rate)
                    }
                    if batchend == len(self.rtriples):
                        _, cur_rel_loss, summary = self.sess.run([self.rel_optimizer, self.rel_loss, self.merged_rel],
                            feed_dict=feed_dict)
                        self.rel_summary_writer.add_summary(summary, iter_num*max_epochs + epoch)
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
                        self.neg_attr_head : self.natriples[i:neg_batchend][:,0].reshape(self.batchSize, self.neg_rel_rate + self.neg_rate),
                        self.neg_val : self.natriples[i:neg_batchend][:,1].reshape(self.batchSize, self.neg_rel_rate + self.neg_rate),
                        self.neg_attr : self.natriples[i:neg_batchend][:,2].reshape(self.batchSize, self.neg_rel_rate + self.neg_rate)
                    }
                    if batchend == len(self.atriples):
                        _, cur_attr_loss, summary = self.sess.run([self.attr_optimizer, self.attr_loss, self.merged_attr],
                            feed_dict=feed_dict)
                        self.attr_summary_writer.add_summary(summary, iter_num*max_epochs + epoch)
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

    def get_top_beta_pairs(self, beta):
        closet_neighbour = [None] * len(self.entity)
        ent_embeddings = self.get_ent_embeddings()
        for (e1,e2) in self.entity_pairs:
            #Compute cosine distance
            distance = abs(spatial.distance.cosine(ent_embeddings[e1], ent_embeddings[e2]))

            #initialize closet neighbour dict.
            if closet_neighbour[e1] is None:
                closet_neighbour[e1] = {'e' : -1, 'd': 100}
            if closet_neighbour[e2] is None:
                closet_neighbour[e2] = {'e' : -1, 'd': 100}

            if closet_neighbour[e1]['d'] > distance:
                closet_neighbour[e1]['e'] = e2
                closet_neighbour[e1]['d'] = distance

            if closet_neighbour[e2]['d'] > distance:
                closet_neighbour[e2]['e'] = e1
                closet_neighbour[e2]['d'] = distance

        aligned_pairs = []
        for (e1,e2) in self.entity_pairs:
            #Skip if no closest neighbour computed
            if closet_neighbour[e1]['e'] == -1:
                continue
            if closet_neighbour[e2]['e'] == -1:
                continue

            if closet_neighbour[e1]['e'] == e2 and closet_neighbour[e2]['e'] == e1:
                aligned_pairs.append((e1, e2))
                if len(aligned_pairs) == beta:
                    break

        return aligned_pairs

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

        self.nrtriples = np.array(get_negative_samples(self.rtriples, len(self.entity),
                                len(self.entity), len(self.relation), self.entity_pairs,
                                neg_rate=self.neg_rate, neg_rel_rate=self.neg_rel_rate))
        return len(self.rtriples)

    def seea_iterate(self, beta=10, max_iter=100, max_epochs=100, swap_relations=False):
        #Initialize tensorflow graph
        with self.sess.as_default():
            self.sess.run(tf.global_variables_initializer())

        predicted_pairs = []
        correct_prediction_count = 0
        for j in range(0, max_iter):

            loss = self.train(max_epochs, j)
            logger.info("Training Complete with loss: %f for iteration: %d", loss, j)

            aligned_pairs = self.get_top_beta_pairs(beta)
            logger.info("Found %d new aligned pairs for iteration: %d", len(aligned_pairs), j)

            if len(aligned_pairs) == 0:
                logger.info("End of SEEA iterations: No new pairs found.")
                return predicted_pairs

            new_triples = [(e1, e2, len(self.relation) - 1) for (e1, e2) in aligned_pairs]
            if swap_relations:
                new_triples.extend(self.swap_parameters(aligned_pairs))
            rtriple_count = self.add_rel_triples(new_triples)
            logger.info("New size of Rtriples %d", rtriple_count)
            predicted_pairs.extend(aligned_pairs)

            #Removed aligned pairs from candidate pairs
            for (e1, e2) in aligned_pairs:
                self.entity_pairs.remove((e1,e2))
                logger.debug("%d, %d aligned. In True pairs: %s", e1, e2, (e1, e2) in self.true_pairs)
                if (e1, e2) in self.true_pairs:
                    correct_prediction_count = correct_prediction_count + 1
            logger.info("%d/%d correctly aligned pairs", correct_prediction_count, len(predicted_pairs))

        logger.info("End of SEEA iterations: Max iteration reached.")
        return predicted_pairs

    def swap_parameters(self, aligned_pairs):
        new_triples = []
        for (a, b) in aligned_pairs:
            at = filter(lambda (h,t,r): h == a, self.rtriples)
            new_triples.extend(map(lambda (h,t,r): (b,t,r), at))

            bt = filter(lambda (h,t,r): h == b, self.rtriples)
            new_triples.extend(map(lambda (h,t,r): (a,t,r), bt))

        return new_triples

    def __del__(self):
        self.close_tf_session()


