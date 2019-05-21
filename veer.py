import config
import tensorflow as tf
import numpy as np

from common import get_logger, get_negative_samples, get_tf_summary_file_path
from scipy import spatial
import timeit

logger = get_logger('RL.VEER')

class VEER(object):
    """
        Value Evolution Embedding for a Record.
        Tensorflow based implementation of VEER method
        Use train method to update the embeddings.
    """

    def __init__(self, dataset, columns, dimension=10, batchSize=100, learning_rate=0.1,
                                    margin=1, regularizer_scale = 0.1):
        """
            Constructor to build the tf model, define required placeholders,
            define loss and optimization method.
        """
        logger.info("Begin generating VEER embeddings with dimension : %d" ,dimension)

        self.dataset = dataset() #Model Class containing data (Census, Cora or FEBRL)
        self.columns = columns #List of column names of interest
        self.dimension = dimension #Embedding Dimension
        self.batchSize = batchSize #BatchSize for Stochastic Gradient Decent
        self.learning_rate = learning_rate #Learning rate for optmizer
        self.margin = margin #margin or bias used for loss computation

        #Collect all values
        self.values = []
        for data in [self.dataset.trainDataA, self.dataset.trainDataB,
                self.dataset.valDataA, self.dataset.valDataB,
                self.dataset.testDataA, self.dataset. testDataB]:
            for col in self.columns:
                self.values.extend(list(data[col]))
        self.values = set(self.values)
        logger.info("No. of unique values: %d", len(self.values))
        self.values = list([self._clean(v) for v in self.values])

        # List of candidate pairs. Remove last incomplete batch if any.
        self.train_candidate = np.array(self.dataset.candidate_links[0: (len(self.dataset.candidate_links)
                                        - len(self.dataset.candidate_links)%batchSize)])
        self.val_candidate = np.array(self.dataset.val_links[0: (len(self.dataset.val_links)
                                        - len(self.dataset.val_links)%batchSize)])
        self.test_candidate = np.array(self.dataset.test_links[0: (len(self.dataset.test_links)
                                        - len(self.dataset.test_links)%batchSize)])

        logger.info("Shape of train candidate: %s", str(self.train_candidate.shape))
        logger.info("Shape of validation candidate: %s", str(self.val_candidate.shape))
        logger.info("Shape of test candidate: %s", str(self.test_candidate.shape))

        #List of ground truth for each candidate pair
        self.train_truth = np.array([1.0 if c in self.dataset.true_links else -1.0
                                        for c in self.train_candidate])
        self.val_truth = np.array([1.0 if c in self.dataset.true_val_links else -1.0
                                        for c in self.val_candidate])
        self.test_truth = np.array([1.0 if c in self.dataset.true_test_links else -1.0
                                        for c in self.test_candidate])

        logger.info("Shape of train truth: %s", str(self.train_truth.shape))
        logger.info("Shape of validation truth: %s", str(self.val_truth.shape))
        logger.info("Shape of test truth: %s", str(self.test_truth.shape))

        # Load values for candidate pairs
        dataA = self.dataset.trainDataA[self.columns]
        dataB = self.dataset.trainDataB[self.columns]
        self.train_records = self._get_records(dataA, dataB, self.train_candidate)

        dataA = self.dataset.valDataA[self.columns]
        dataB = self.dataset.valDataB[self.columns]
        self.val_records = self._get_records(dataA, dataB, self.val_candidate)

        dataA = self.dataset.testDataA[self.columns]
        dataB = self.dataset.testDataB[self.columns]
        self.test_records = self._get_records(dataA, dataB, self.test_candidate)

        logger.info("Shape of train records: %s", str(self.train_records.shape))
        logger.info("Shape of validation records: %s", str(self.val_records.shape))
        logger.info("Shape of test records: %s", str(self.test_records.shape))

        #Define Embedding Variables
        initializer = tf.contrib.layers.xavier_initializer(uniform = True)
        regularizer = tf.contrib.layers.l2_regularizer(scale = regularizer_scale)

        self.val_embeddings = tf.get_variable(name = "val_embeddings",
                                    shape = [len(self.values), dimension],
                                    initializer = initializer, regularizer = regularizer)
        self.col_weights = tf.get_variable(name = "col_weights",
                                    shape = [len(self.columns)],
                                    initializer = initializer, regularizer = regularizer)

        #Define Placeholders for input
        self.record_a = tf.placeholder(tf.int32, shape=[self.batchSize, len(self.columns)])
        self.record_b = tf.placeholder(tf.int32, shape=[self.batchSize, len(self.columns)])
        self.truth_val = tf.placeholder(tf.float32, shape=[self.batchSize, 1])

        pos_a = tf.nn.embedding_lookup(self.val_embeddings, self.record_a)
        pos_b = tf.nn.embedding_lookup(self.val_embeddings, self.record_b)
        pos_col_wt = tf.nn.embedding_lookup(self.col_weights, range(len(self.columns)))

        logger.info("Shape of pos a :%s", str(pos_a.shape))
        logger.info("Shape of pos b :%s", str(pos_b.shape))
        logger.info("Shape of pos col wt :%s", str(pos_col_wt.shape))

        #Normalize embeddings
        pos_a = tf.nn.l2_normalize(pos_a, 2)
        pos_b = tf.nn.l2_normalize(pos_b, 2)
        pos_col_wt = tf.nn.l2_normalize(pos_col_wt)

        #Compute loss and prediction
        self.score = tf.matmul(tf.reduce_mean(tf.abs(pos_a - pos_b), 2, keepdims=False),
                         tf.expand_dims(pos_col_wt, 1))
        logger.info("Shape of score: %s", str(self.score.shape))
        self.predict = tf.sigmoid(self.score)
        logger.info("Shape of predict: %s", str(self.predict.shape))
        _loss = tf.maximum(self.margin + (self.score * self.truth_val), 0)
        logger.info("Shape of _loss: %s", str(_loss.shape))
        self.loss = tf.reduce_sum(tf.reduce_mean(_loss, 1, keepdims = False), keepdims = True)
        logger.info("Shape of loss: %s", str(self.loss.shape))

        logger.info("Aggregated loss shape:%s", str(self.loss[0].shape))

        #collect summary parameters
        tf.summary.scalar('loss', self.loss[0])

        #Configure Performance measures
        int_truth_val = tf.cast((self.truth_val+1)/2, tf.int64)
        int_predict_val = tf.cast(self.predict, tf.int64)

        self.accuracy = tf.contrib.metrics.accuracy(int_predict_val, int_truth_val)
        tf.summary.scalar('Accuracy', self.accuracy)
        """
        #Todo: F-Score is always 0, fix it.
        self.f1_score, _ = tf.contrib.metrics.f1_score(int_truth_val, self.predict[:,0])
        tf.summary.scalar('F-Score', self.f1_score)

        #Todo: MAP@1 is not as per the queries we expect based on match / non-match class.
        int_predict_val = tf.reshape(tf.stack([self.predict, 1 - self.predict], 1),
                                (self.batchSize, 2))
        logger.info("Shape of int_predict_val: %s", str(int_predict_val.shape))
        self.map1, _ = tf.metrics.average_precision_at_k(int_truth_val, int_predict_val, 1)
        self.map10, _ = tf.metrics.average_precision_at_k(int_truth_val, int_predict_val, 2)
        tf.summary.scalar('MAP@1', self.map1)
        tf.summary.scalar('MAP@10', self.map10)
        """

        #Configure optimizer
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

        #Configure session
        self.sess = tf.Session()

        #Confirgure summary location
        self.merged = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(get_tf_summary_file_path(logger) + '/train', self.sess.graph)
        self.validation_summary_writer = tf.summary.FileWriter(get_tf_summary_file_path(logger) + '/val',
                                    self.sess.graph)

    def _get_records(self, dataA, dataB, candidates):
        """
            Private method to map vaules to its index for each record
        """
        dataA_dict = {id : [self.values.index(self._clean(r[i]))
                                for i in range(len(self.columns))]
                                    for id, r in dataA.iterrows()}
        dataB_dict = {id : [self.values.index(self._clean(r[i]))
                                for i in range(len(self.columns))]
                                    for id, r in dataB.iterrows()}
        records = [(dataA_dict[a], dataB_dict[b]) for (a, b) in candidates]
        return np.array(records)

    def _clean(self, val):
        """
            Private method to clean and standardize the value as string
        """
        return str(unicode(val).encode('utf-8', 'ignore')).strip().lower()

    def train(self, max_epochs=100):
        """
            Method to train the model and update the embeddings.
        """
        loss = 0
        with self.sess.as_default():
            self.sess.run(tf.local_variables_initializer())
            self.sess.run(tf.global_variables_initializer())
            logger.info("Local Vars: %s", str([i for i in tf.local_variables()]))

            for epoch in range(0, max_epochs):
                #Training
                loss = 0
                accuracy = 0
                batch_starts = np.arange(0, len(self.train_records), self.batchSize)
                np.random.shuffle(batch_starts)
                for i in batch_starts:
                    batchend = min(len(self.train_records), i + self.batchSize)
                    feed_dict = {
                        self.record_a : self.train_records[i:batchend][:, 0].reshape(
                                                    self.batchSize, len(self.columns)),
                        self.record_b : self.train_records[i:batchend][:, 1].reshape(
                                                    self.batchSize, len(self.columns)),
                        self.truth_val  : self.train_truth[i:batchend].reshape(self.batchSize, 1),
                        }

                    if batchend ==  len(self.train_records):
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
                batch_starts = np.arange(0, len(self.val_records), self.batchSize)
                np.random.shuffle(batch_starts)
                for i in batch_starts:
                    batchend = min(len(self.val_records), i + self.batchSize)
                    feed_dict = {
                        self.record_a : self.val_records[i:batchend][:, 0].reshape(
                                                        self.batchSize, len(self.columns)),
                        self.record_b : self.val_records[i:batchend][:, 1].reshape(
                                                        self.batchSize, len(self.columns)),
                        self.truth_val : self.val_truth[i:batchend].reshape(self.batchSize, 1)
                        }

                    if batchend == len(self.val_records):
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
        """
            Method to generate predictions over test data
        """
        predict = []
        accuracy = 0
        with self.sess.as_default():
            batch_starts = np.arange(0, len(self.test_records), self.batchSize)
            for i in batch_starts:
                batchend = min(len(self.test_records), i + self.batchSize)
                feed_dict = {
                    self.record_a : self.test_records[i:batchend][:, 0].reshape(
                                                    self.batchSize, len(self.columns)),
                    self.record_b : self.test_records[i:batchend][:, 1].reshape(
                                                    self.batchSize, len(self.columns)),
                    self.truth_val : self.test_truth[i:batchend].reshape(self.batchSize, 1)
                    }

                cur_predict, cur_accuracy = self.sess.run([self.predict, self.accuracy],
                                                            feed_dict=feed_dict)
                predict.extend([float(p) for p in list(cur_predict)])
                accuracy = accuracy + cur_accuracy

            predict = [(self.test_candidate[i][0], self.test_candidate[i][1], predict[i])
                                for i in range(len(predict))]
            accuracy = accuracy / len(batch_starts)
        return (predict, accuracy)

    def get_val_embeddings(self):
        with self.sess.as_default():
            return tf.nn.embedding_lookup(self.val_embeddings, range(0, len(self.values))).eval()

    def get_col_weights(self):
        with self.sess.as_default():
            return tf.nn.embedding_lookup(self.col_weights, range(0, len(self.columns))).eval()

    def close_tf_session(self):
        tf.reset_default_graph()
        self.sess.close()

    def __str__(self):
        return "VEER"

    def __del__(self):
        self.close_tf_session()
