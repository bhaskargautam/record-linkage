import config
import logging
import numpy as np
import random
import recordlinkage
import pandas as pd
import sys

def get_logger(name, filename=config.DEFAULT_LOG_FILE, level=logging.DEBUG):
    log_file_path = config.BASE_OUTPUT_FOLDER + filename
    formatter = logging.Formatter('%(name)s %(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger(name)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(filename=log_file_path)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(file_handler)
    else:
        logger.debug("Handlers already exist: %s", str(logger.handlers))

    logging.basicConfig(level=level,
                        format='%(name)s %(asctime)s %(levelname)s %(message)s')
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

def get_negative_samples(triples, total_head, total_tail, total_rel,
                            entity_pairs, neg_rate=1, neg_rel_rate=0):
    """
        Method for sampling negative triplets.
        Randomly repalces head, tail or relation with equal probability.
        Number of negative samples = len(triples) * neg_rate + len(triples) * neg_rel_rate
    """
    ntriples = []
    all_head_index = xrange(0, total_head)
    all_tail_index = xrange(0, total_tail)
    all_rel_index = xrange(0, total_rel)
    tuple_triples = set(map(tuple, triples))
    tuple_ep = set(map(tuple, entity_pairs))
    logger = get_logger("RL.NEG_ATTR_SAMPLER")

    for (h, t, r) in triples:
        #logger.debug("Finding neg sample for %d, %d, %d", h, t, r)
        for neg_iter in range(0, neg_rate):
            rand_choice = random.randint(0,1)
            if(rand_choice == 0):
                #replace head with neg_head
                while True:
                    neg_head = random.choice(all_head_index)
                    if neg_head == h or (neg_head, t) in tuple_ep or \
                            (t, neg_head) in tuple_ep:
                        continue
                    if (neg_head, t, r) not in tuple_triples:
                        ntriples.append((neg_head, t, r))
                        break
                #logger.debug("Replaced head: %s", str(ntriples[-1]))
            else:
                #replace tail with neg_tail
                while True:
                    neg_tail = random.choice(all_tail_index)
                    if neg_tail == t or (h, neg_tail) in tuple_ep or \
                            (neg_tail, h) in tuple_ep:
                        continue
                    if (h, neg_tail, r) not in tuple_triples:
                        ntriples.append((h, neg_tail, r))
                        break
                #logger.debug("Replaced tail: %s", str(ntriples[-1]))
        for neg_iter in range(0, neg_rel_rate):
            #replace rel with neg_rel
            while True:
                neg_rel = random.choice(all_rel_index)
                if neg_rel == r:
                    continue
                if (h, t, neg_rel) not in tuple_triples:
                    ntriples.append((h, t, neg_rel))
                    break
            #logger.debug("Replaced rel: %s", str(ntriples[-1]))

    logger.info("Number of negative triples: %d", len(ntriples))
    return ntriples

def export_embeddings(graph_type, model, method, entity, ent_emebedding):
    base_file_name = config.BASE_OUTPUT_FOLDER + str(graph_type) +  "/"
    base_file_name = base_file_name + str(model) + "_" + str(method)
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
    return True

def export_result_prob(dataset, graph_type, dataset_prefix, method,
                            entity, result_prob, true_pairs):
    base_file_name = config.BASE_OUTPUT_FOLDER + str(graph_type) +  "/"
    base_file_name = base_file_name + str(dataset_prefix) + "_" + str(method)
    result_prob = sorted(result_prob, key=lambda x: x[2])
    false_positives = []
    with open(base_file_name + "_result_prob.tsv", "w+") as f:
        for (e1, e2, d) in result_prob:
            f.write("%s\t%s\t%f\t%s\n" % (entity[e1], entity[e2], d, (e1, e2) in true_pairs))
            if (e1, e2) not in true_pairs and \
                    len(false_positives) < config.MAX_FALSE_POSITIVE_TO_LOG:
                false_positives.append((e1, e2))

    #Log full information about top False Positives
    model = dataset()
    with open(base_file_name + "_false_positives.txt", "w+") as f:
        for (e1, e2) in false_positives:
            f.write("\nRecord A:\n%s\n" % str(model.get_entity_information(entity[e1])))
            f.write("Record B:\n%s\n" % str(model.get_entity_information(entity[e2])))

    return True

def get_optimal_threshold(result_prob, true_pairs):
    logger = get_logger('RL.OPTIMAL_THRESHOLD')
    max_fscore = 0.0
    optimal_threshold = 0
    for threshold in range(20,110, 5):
        threshold = threshold / 100.0
        try:
            result = pd.MultiIndex.from_tuples([(e1, e2) for (e1, e2, d) in result_prob if d <= threshold])
            fscore = recordlinkage.fscore(true_pairs, result)
            logger.info("For threshold: %f fscore: %f", threshold, fscore)
            if fscore >= max_fscore:
                max_fscore = fscore
                optimal_threshold = threshold
        except Exception as e:
            logger.error(e)
            continue
    logger.info("Found optimal threshold %f with max_fscore: %f", optimal_threshold, max_fscore)
    return (optimal_threshold, max_fscore)

### INFORMATION RETRIEVAL METRICS

def get_precision_at_k(result_prob, true_pairs, k=1):
    result_prob = sorted(result_prob, key=(lambda x: x[2]))
    true_results = [(e1, e2) for (e1,e2,d) in result_prob[:k] if (e1, e2) in true_pairs]
    return (len(true_results) * 1.0) / k

def get_average_precision(result_prob, true_pairs):
    result_prob = sorted(result_prob, key=(lambda x: x[2]))
    k = 0
    total_precision = 0.0
    for i in range(1, len(true_pairs) + 1):
        while k < len(result_prob):
            if (result_prob[k][0], result_prob[k][1]) in true_pairs:
                total_precision = total_precision + (float(i) / (k+1))
                k = k + 1
                break
            else:
                k = k + 1

    if len(true_pairs):
        return total_precision / float(len(true_pairs))
    return 0

def get_mean_reciprocal_rank(result_prob, true_pairs):
    reciprocal_rank_sum = 0.0
    tp_map = {}
    for e1, e2 in true_pairs:
        if e1 in tp_map:
            tp_map[e1].append(e2)
        else:
            tp_map[e1] = [e2]

    for e1 in tp_map:
        results = [(h,t,d) for (h,t,d) in result_prob if h == e1]
        results = sorted(results, key=lambda x: x[2])
        for i in range(0,len(results)):
            if results[i][1] in tp_map[e1]:
                reciprocal_rank_sum = reciprocal_rank_sum + 1.0/(i + 1)
                break

    if len(tp_map):
        return reciprocal_rank_sum / len(tp_map)
    return 0

def get_mean_rank(result_prob, true_pairs):
    rank_sum = 0
    for (e1, e2) in true_pairs:
        results = [(h,t,d) for (h,t,d) in result_prob if h == e1]
        results = sorted(results, key=lambda x: x[2])
        for i in range(0,len(results)):
            if results[i][1] == e2:
                rank_sum = rank_sum + i + 1

    if len(true_pairs):
        return rank_sum / float(len(true_pairs))
    return 0

def get_mean_average_precision(result_prob, true_pairs):
    average_precision = 0.0
    tp_map = {}
    for e1, e2 in true_pairs:
        if e1 in tp_map:
            tp_map[e1].append(e2)
        else:
            tp_map[e1] = [e2]

    for e1 in tp_map:
        total_precision = 0.0
        true_count = 0
        results = [(h,t,d) for (h,t,d) in result_prob if h == e1]
        results = sorted(results, key=lambda x: x[2])
        for i in range(0, len(results)):
            if results[i][1] in tp_map[e1]:
                true_count = true_count + 1
                total_precision = total_precision + float(true_count)/(i+1)

        average_precision = average_precision + (total_precision/float(len(tp_map[e1])))

    if len(tp_map):
        return average_precision / len(tp_map)
    return 0

def log_ir_metrics(logger, result_prob, true_pairs):
    logger.info("Precision@1 = %f", get_precision_at_k(result_prob, true_pairs, k=1))
    logger.info("Precision@5 = %f", get_precision_at_k(result_prob, true_pairs, k=5))
    logger.info("Precision@10 = %f", get_precision_at_k(result_prob, true_pairs, k=10))
    logger.info("Precision@20 = %f", get_precision_at_k(result_prob, true_pairs, k=20))

    logger.info("Average Precision (AP) = %f", get_average_precision(result_prob, true_pairs))
    logger.info("Mean Rank (MR)= %f", get_mean_rank(result_prob, true_pairs))
    logger.info("Mean Reciprocal Rank (MRR)= %f", get_mean_reciprocal_rank(result_prob, true_pairs))
    logger.info("Mean Average Precision (MAP)= %f", get_mean_average_precision(result_prob, true_pairs))
    return True