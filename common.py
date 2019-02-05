import logging
import numpy as np
import recordlinkage

def get_logger(name, filename='test.log', level=logging.DEBUG):
    logging.basicConfig(filename=filename, level=level,
        format='%(name)s %(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger(name)
    logger.addHandler(logging.StreamHandler())
    return logger

def write_results(results_for, fscore, accuracy, precision, recall):
    f = open('result.log', 'a+')
    f.write("%s, %f, %f, %f, %f\n" % (results_for, fscore, accuracy, precision, recall))
    f.close()

def log_quality_results(logger, result, true_links, total_pairs):
    logger.info("Number of Results %d", len(result))
    logger.info("Confusion Matrix %s", str(
        recordlinkage.confusion_matrix(true_links, result, total_pairs)))
    try:
        fscore = recordlinkage.fscore(true_links, result)
        accuracy = recordlinkage.accuracy(true_links, result, total_pairs)
        precision = recordlinkage.precision(true_links, result)
        recall = recordlinkage.recall(true_links, result)
        logger.info("FScore: %f Accuracy : %f", fscore, accuracy)
        logger.info("Precision: %f Recall %f", precision, recall)
        write_results(logger.name, fscore, accuracy, precision, recall)

    except ZeroDivisionError:
        logger.error("ZeroDivisionError!!")


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))
