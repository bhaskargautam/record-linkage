import config
import logging
import numpy as np
import random
import recordlinkage
import pandas as pd

def get_logger(name, filename=config.DEFAULT_LOG_FILE, level=logging.DEBUG):
    log_file_path = config.BASE_OUTPUT_FOLDER + filename
    logging.basicConfig(filename=log_file_path, level=level,
        format='%(name)s %(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger(name)
    logger.addHandler(logging.StreamHandler())
    return logger

def write_results(results_for, fscore, accuracy, precision, recall, params):
    with open(config.BASE_OUTPUT_FOLDER + config.DEFAULT_RESULT_LOG_FILE, 'a+') as f:
        f.write("%s, %f, %f, %f, %f, %s\n" % (results_for, fscore, accuracy,
            precision, recall, str(params)))

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

def export_embeddings(model, method, entity, ent_emebedding):
    base_file_name = config.BASE_OUTPUT_FOLDER + str(model) + "_" + str(method)
    with open(base_file_name + "_embedding.tsv", "w+") as f:
        for e in ent_emebedding:
            for i in range(0, len(e)):
                f.write("%f\t" % e[i])
            f.write("\n")

    with open(base_file_name + "_meta.tsv", "w+") as f:
        for e in entity:
            try:
                f.write("%s\n" % str(e))
            except UnicodeEncodeError:
                f.write("%s\n" % str(e.encode('ascii', 'ignore').decode('ascii')))

def get_optimal_threshold(result_prob, true_pairs):
    logger = get_logger('OPTIMAL_THRESHOLD')
    max_fscore = 0.0
    optimal_threshold = 0
    for threshold in range(20,110, 5):
        threshold = threshold / 100.0
        try:
            result = pd.MultiIndex.from_tuples([(e1, e2) for (e1, e2, d) in result_prob if d <= threshold])
            fscore = recordlinkage.fscore(true_pairs, result)
            if fscore >= max_fscore:
                max_fscore = fscore
                optimal_threshold = threshold
        except Exception as e:
            logger.error(e)
            continue
    logger.info("Found optimal threshold %f with max_fscore: %f", optimal_threshold, max_fscore)
    return (optimal_threshold, max_fscore)