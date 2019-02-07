import logging
import numpy as np
import random
import recordlinkage

def get_logger(name, filename='test.log', level=logging.DEBUG):
    logging.basicConfig(filename=filename, level=level,
        format='%(name)s %(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger(name)
    logger.addHandler(logging.StreamHandler())
    return logger

def write_results(results_for, fscore, accuracy, precision, recall, params):
    f = open('result.log', 'a+')
    f.write("%s, %f, %f, %f, %f, %s\n" % (results_for, fscore, accuracy, precision, recall, str(params)))
    f.close()

def log_quality_results(logger, result, true_links, total_pairs, params=None):
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
        logger.info("For params : %s", str(params))
        write_results(logger.name, fscore, accuracy, precision, recall, params)

    except ZeroDivisionError:
        logger.error("ZeroDivisionError!!")

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def get_negative_samples(triples, total_head, total_tail, total_rel, neg_rate=1):
    """
        Method for sampling negative triplets.
        Randomly repalces head, tail or relation with equal probability.
        Number of negative samples = len(triples) * neg_rate
    """
    ntriples = []
    all_head_index = range(0, total_head)
    all_tail_index = range(0, total_tail)
    all_rel_index = range(0, total_rel)
    tuple_triples = set(map(tuple, triples))
    logger = get_logger("NEG_ATTR_SAMPLER")

    for (h, t, r) in triples:
        #logger.debug("Finding neg sample for %d, %d, %d", h, t, r)
        for neg_iter in range(0, neg_rate):
            np.random.shuffle(all_head_index)
            np.random.shuffle(all_rel_index)
            np.random.shuffle(all_tail_index)
            rand_choice = random.randint(0,2)
            if(rand_choice == 0):
                #replace head with neg_head
                for neg_head in all_head_index:
                    if neg_head == h:
                        continue
                    if (neg_head, t, r) not in tuple_triples:
                        ntriples.append((neg_head, t, r))
                        break
                #logger.debug("Replaced head: %s", str(ntriples[-1]))
            elif(rand_choice == 1):
                #replace tail with neg_tail
                for neg_tail in all_tail_index:
                    if neg_tail == t:
                        continue
                    if (h, neg_tail, r) not in tuple_triples:
                        ntriples.append((h, neg_tail, r))
                        break
                #logger.debug("Replaced tail: %s", str(ntriples[-1]))
            else:
                #replace rel with neg_rel
                for neg_rel in all_rel_index:
                    if neg_rel == r:
                        continue
                    if (h, t, neg_rel) not in tuple_triples:
                        ntriples.append((h, t, neg_rel))
                        break
                #logger.debug("Replaced rel: %s", str(ntriples[-1]))

    logger.info("Number of negative triples: %d", len(ntriples))
    return ntriples