import config
import tensorflow as tf
import numpy as np

from common import get_logger, get_negative_samples, get_tf_summary_file_path
from scipy import spatial
import timeit

logger = get_logger('RL.WERL')

class WERL(object):
    "Weigted Embedding based Record Linkage"

    def __init__(self, model, columns, entity, ent_embeddings, rel_embedding=None,
                                    batchSize=100, learning_rate=0.1,
                                    margin=1, regularizer_scale = 0.1):
        logger.info("Initializing WERL to learn weights")
        self.dataset = model()
        self.columns = columns
        self.entity = entity
        self.ent_embeddings = ent_embeddings
        self.dimension = len(ent_embeddings[0])
        self.batchSize = batchSize
        self.learning_rate = learning_rate
        self.margin = margin
        self.regularizer_scale = regularizer_scale

        self.map_ent_to_embedding = {entity[i] : ent_embeddings[i] for i in range(len(entity))}
        self.zero_vector = [1] * self.dimension
        if rel_embedding is None:
            self.rel_embedding = np.zeros((len(self.columns), self.dimension))
        else:
            self.rel_embedding = rel_embedding
        self.get_embed = (lambda x: self.map_ent_to_embedding[x]
                                if x in self.map_ent_to_embedding else self.zero_vector)

        #Colect Train Data
        self.train_records = [(self.dataset.trainDataA.loc[a], self.dataset.trainDataB.loc[b])
                                    for (a, b) in self.dataset.candidate_links]
        self.train_same_val = np.array([[0 if a[c] == b[c] else 1 for c in columns]
                                for (a, b) in self.train_records])
        self.train_records = np.array([([self.get_embed(a[c]) for c in columns],
                                [self.get_embed(b[c]) for c in columns])
                                for (a, b) in self.train_records])
        self.train_truth = np.array([1 if cp in self.dataset.true_links else -1
                                    for cp in self.dataset.candidate_links])

        #Colect Validation Data
        self.val_records = [(self.dataset.valDataA.loc[a], self.dataset.valDataB.loc[b])
                                    for (a, b) in self.dataset.val_links]
        self.val_same_val = np.array([[0 if a[c] == b[c] else 1 for c in columns]
                                for (a, b) in self.val_records])
        self.val_records = np.array([([self.get_embed(a[c]) for c in columns],
                                [self.get_embed(b[c]) for c in columns])
                                for (a, b) in self.val_records])
        self.val_truth = np.array([1 if cp in self.dataset.true_val_links else -1
                                    for cp in self.dataset.val_links])

        #Colect Test Data
        self.test_records = [(self.dataset.testDataA.loc[a], self.dataset.testDataB.loc[b])
                                    for (a, b) in self.dataset.test_links]
        self.test_same_val = np.array([[0 if a[c] == b[c] else 1 for c in columns]
                                for (a, b) in self.test_records])
        self.test_records = np.array([([self.get_embed(a[c]) for c in columns],
                                [self.get_embed(b[c]) for c in columns])
                                for (a, b) in self.test_records])
        self.test_truth = np.array([1 if cp in self.dataset.true_test_links else -1
                                        for cp in self.dataset.test_links])

        #Define Trainable Weights for each feature
        initializer = tf.contrib.layers.xavier_initializer(uniform = True)
        regularizer = tf.contrib.layers.l2_regularizer(scale = regularizer_scale)
        self.weights = tf.get_variable(name = "weights", shape = [len(self.columns), 1], #self.dimension],
                                    initializer = initializer, regularizer = regularizer)

        #Define Placeholders for input
        self.record_a = tf.placeholder(tf.float32, shape=[self.batchSize, len(columns), self.dimension])
        self.record_b = tf.placeholder(tf.float32, shape=[self.batchSize, len(columns), self.dimension])
        self.truth_val = tf.placeholder(tf.float32, shape=[self.batchSize, 1])
        self.same_val = tf.placeholder(tf.float32, shape=[self.batchSize, len(columns)])
        norm_weights = tf.nn.l2_normalize(self.weights)

        long_val = tf.tile(tf.expand_dims(self.same_val, 2), [1, 1, self.dimension])
        logger.info("long_val %s", str(long_val.shape))

        self.score_merl = tf.reduce_sum(tf.math.multiply(
                        tf.abs(self.record_a - self.record_b + self.rel_embedding),
                        long_val), 2, keepdims=False) / len(self.columns)
        self.score = tf.matmul(self.score_merl, norm_weights)
        self.predict = tf.sigmoid(self.score) #/ len(self.columns))
        self.predict_merl = tf.sigmoid(self.score_merl) #/ len(self.columns))
        _loss = tf.maximum(0.0, self.margin + (self.score * self.truth_val))
        self.loss = tf.reduce_sum(tf.reduce_mean(_loss, 1, keepdims = False), keepdims = True)
        #_loss = tf.reduce_sum(tf.reduce_mean((self.score * self.truth_val), 1, keepdims = False), keepdims = True)
        #self.loss = tf.maximum(0.0, margin + _loss)
        #collect summary parameters
        tf.summary.scalar('loss', self.loss[0])

        #Configure Performance measures
        int_truth_val = tf.cast((self.truth_val+1)/2, tf.int64)
        int_predict_val = tf.cast(self.predict, tf.int64)

        self.accuracy = tf.contrib.metrics.accuracy(int_predict_val, int_truth_val)
        tf.summary.scalar('Accuracy', self.accuracy)

        #Configure optimizer
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

        #Configure session
        self.sess = tf.Session()

        #Confirgure summary location
        self.merged = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(get_tf_summary_file_path(logger) + '/train', self.sess.graph)
        self.validation_summary_writer = tf.summary.FileWriter(get_tf_summary_file_path(logger) + '/val',
                                    self.sess.graph)

        #Configure Saver
        self.saver = tf.train.Saver()

    def train(self, max_epochs=100):
        #Remove last partial batch if any
        last_index = self.train_records.shape[0] - (self.train_records.shape[0]%self.batchSize)
        self.train_records = self.train_records[0:last_index]
        self.train_truth = self.train_truth[0:last_index]
        logger.info("Shape of train_records %s", str(self.train_records.shape))

        #Remove last partial batch if any
        last_index = self.val_records.shape[0] - (self.val_records.shape[0]%self.batchSize)
        self.val_records = self.val_records[0:last_index]
        self.val_truth = self.val_truth[0:last_index]
        logger.info("Shape of val_records %s", str(self.val_records.shape))

        loss = 0
        with self.sess.as_default():
            self.sess.run(tf.local_variables_initializer())
            self.sess.run(tf.global_variables_initializer())
            logger.info("Local Vars: %s", str([i for i in tf.local_variables()]))

            for epoch in range(0, max_epochs):
                #Training
                loss = 0
                accuracy = 0
                batch_starts = np.arange(0, self.train_records.shape[0], self.batchSize)
                np.random.shuffle(batch_starts)
                for i in batch_starts:
                    batchend = min(self.train_records.shape[0], i + self.batchSize)
                    feed_dict = {
                        self.record_a : self.train_records[i:batchend][:, 0].reshape(self.batchSize,
                                                len(self.columns), self.dimension),
                        self.record_b : self.train_records[i:batchend][:, 1].reshape(self.batchSize,
                                                len(self.columns), self.dimension),
                        self.truth_val  : self.train_truth[i:batchend].reshape(self.batchSize, 1),
                        self.same_val : self.train_same_val[i:batchend]
                        }

                    if batchend ==  self.train_records.shape[0]:
                        _, cur_loss, cur_accuracy, summary = self.sess.run([self.optimizer,
                                self.loss, self.accuracy, self.merged], feed_dict=feed_dict)
                        self.summary_writer.add_summary(summary, epoch)
                    else:
                        #train network on batch
                        _, cur_loss, cur_accuracy = self.sess.run([self.optimizer, self.loss,
                                self.accuracy], feed_dict=feed_dict)

                    if type(cur_loss) == list:
                        cur_loss = cur_loss[0]
                    loss = loss + cur_loss
                    accuracy = accuracy + cur_accuracy
                accuracy = accuracy / len(batch_starts)
                if loss:
                    logger.info("Epoch: %d Loss: %f Accuracy: %f", epoch, loss, accuracy)
                else:
                    logger.info("Zero Loss, finish training in %d epochs", epoch)
                    break

                #Validation
                val_loss = 0
                accuracy = 0
                batch_starts = np.arange(0, self.val_records.shape[0], self.batchSize)
                np.random.shuffle(batch_starts)
                for i in batch_starts:
                    batchend = min(self.val_records.shape[0], i + self.batchSize)
                    feed_dict = {
                        self.record_a : self.val_records[i:batchend][:, 0].reshape(
                                            self.batchSize, len(self.columns), self.dimension),
                        self.record_b : self.val_records[i:batchend][:, 1].reshape(
                                            self.batchSize, len(self.columns), self.dimension),
                        self.truth_val : self.val_truth[i:batchend].reshape(self.batchSize, 1),
                        self.same_val : self.val_same_val[i:batchend]
                        }

                    if batchend == self.val_records.shape[0]:
                        cur_loss, cur_accuracy, summary = self.sess.run([ self.loss,
                                    self.accuracy, self.merged], feed_dict=feed_dict)
                        self.validation_summary_writer.add_summary(summary, epoch)
                    else:
                        #validate network on batch
                        cur_loss, cur_accuracy = self.sess.run([self.loss, self.accuracy],
                                            feed_dict=feed_dict)
                    if type(cur_loss) == list:
                        cur_loss = cur_loss[0]
                    val_loss = val_loss + cur_loss
                    accuracy = accuracy + cur_accuracy
                accuracy = accuracy / len(batch_starts)
                if val_loss:
                    logger.info("Epoch: %d VAL Loss: %f Accuracy: %f", epoch, val_loss, accuracy)
                else:
                    logger.info("Zero VAL Loss, finish training in %d epochs", epoch)
                    break
        return (loss, val_loss)

    def test(self):
        #Remove last partial batch if any
        last_index = self.test_records.shape[0] - (self.test_records.shape[0]%self.batchSize)
        self.test_records = self.test_records[0:last_index]
        self.test_truth = self.test_truth[0:last_index]
        logger.info("Shape of test_records %s", str(self.test_records.shape))

        predict = []
        accuracy = 0
        with self.sess.as_default():
            batch_starts = np.arange(0, self.test_records.shape[0], self.batchSize)
            for i in batch_starts:
                batchend = min(self.test_records.shape[0], i + self.batchSize)
                feed_dict = {
                    self.record_a : self.test_records[i:batchend][:, 0].reshape(self.batchSize,
                                            len(self.columns), self.dimension),
                    self.record_b : self.test_records[i:batchend][:, 1].reshape(self.batchSize,
                                            len(self.columns), self.dimension),
                    self.truth_val : self.test_truth[i:batchend].reshape(self.batchSize, 1),
                    self.same_val : self.test_same_val[i:batchend]
                    }

                cur_predict, cur_accuracy = self.sess.run([self.predict, self.accuracy],
                                                            feed_dict=feed_dict)
                predict.extend([float(p) for p in list(cur_predict)])
                accuracy = accuracy + cur_accuracy

            predict = [(self.dataset.test_links[i][0], self.dataset.test_links[i][1],
                            predict[i]) for i in range(len(predict))]

            accuracy = accuracy / len(batch_starts)
        return (predict, accuracy)

    def test_merl(self):
        #Remove last partial batch if any
        last_index = self.test_records.shape[0] - (self.test_records.shape[0]%self.batchSize)
        self.test_records = self.test_records[0:last_index]
        self.test_truth = self.test_truth[0:last_index]
        logger.info("Shape of test_records %s", str(self.test_records.shape))

        predict = []
        accuracy = 0
        with self.sess.as_default():
            batch_starts = np.arange(0, self.test_records.shape[0], self.batchSize)
            #self.weights = self.weights.assign(np.ones((len(self.columns), 1)))
            for i in batch_starts:
                batchend = min(self.test_records.shape[0], i + self.batchSize)
                feed_dict = {
                    self.record_a : self.test_records[i:batchend][:, 0].reshape(self.batchSize,
                                            len(self.columns), self.dimension),
                    self.record_b : self.test_records[i:batchend][:, 1].reshape(self.batchSize,
                                            len(self.columns), self.dimension),
                    self.truth_val : self.test_truth[i:batchend].reshape(self.batchSize, 1),
                    self.same_val : self.test_same_val[i:batchend]
                    }

                cur_predict, cur_accuracy = self.sess.run([self.predict_merl, self.accuracy],
                                                            feed_dict=feed_dict)
                predict.extend([float(sum(p)/len(p)) for p in list(cur_predict)])
                accuracy = accuracy + cur_accuracy

            predict = [(self.dataset.test_links[i][0], self.dataset.test_links[i][1],
                            predict[i]) for i in range(len(predict))]

            accuracy = accuracy / len(batch_starts)
        return (predict, accuracy)


    def test_without_weight(self):
        """Mean Embedding based RL TEST"""
        #Remove last partial batch if any
        last_index = self.test_records.shape[0] - (self.test_records.shape[0]%self.batchSize)
        self.test_records = self.test_records[0:last_index]
        self.test_truth = self.test_truth[0:last_index]
        logger.info("Shape of test_records %s", str(self.test_records.shape))

        with self.sess.as_default():
            #self.sess.run(tf.local_variables_initializer())
            #self.sess.run(tf.global_variables_initializer())
            logger.info("Testing without using the WERL weights:")
            predict = [sum([ 0 if np.array_equal(a[i], b[i]) else abs(
                            a[i] + self.rel_embedding - b[i])
                                for i in range(len(a))])/len(a)
                                    for (a, b) in self.test_records]
            predict = [np.mean(p) for p in predict]
            accuracy = [1 if self.test_truth[i] == (predict[i] <= 0.5) else 0 for i in range(len(self.test_truth))]
            accuracy = sum(accuracy) / float(len(self.test_truth))
            logger.info("Accuracy %.2f", accuracy)

            predict = [(self.dataset.test_links[i][0], self.dataset.test_links[i][1],
                                predict[i]) for i in range(len(predict))]
            return (predict, accuracy)

    def get_col_weights(self):
        with self.sess.as_default():
            return tf.nn.embedding_lookup(self.weights, range(0, len(self.columns))).eval()

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
        return "WERL"
