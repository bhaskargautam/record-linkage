import logging
import recordlinkage

def get_logger(name):
    logging.basicConfig(filename='test.log',level=logging.DEBUG,
        format='%(name)s %(asctime)s %(levelname)s %(message)s')
    return logging.getLogger(name)

def log_quality_results(logger, result, true_links, total_pairs):
    logger.info("Number of Results %d", len(result))
    logger.info("Confusion Matrix %s", str(
        recordlinkage.confusion_matrix(true_links, result, total_pairs)))
    logger.info("FScore: %f", recordlinkage.fscore(true_links, result))
    logger.info("Training Accuracy: %f", recordlinkage.accuracy(true_links, result, total_pairs))
    logger.info("Precision: %f", recordlinkage.precision(true_links, result))
    logger.info("Recall: %f", recordlinkage.recall(true_links, result))