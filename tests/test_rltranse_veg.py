import unittest
import itertools
import pandas as pd
import numpy as np
import recordlinkage
import timeit
import unittest

from common import (
    export_embeddings,
    export_false_positives,
    export_false_negatives,
    export_human_readable_results,
    export_result_prob,
    get_optimal_threshold,
    get_logger,
    InformationRetrievalMetrics,
    log_quality_results,
    sigmoid)
from data.base_census import CensusFields, census_field_map, CensusLocation
from data.census import Census
from data.cora import Cora
from data.febrl import FEBRL
from VEG.rltranse import RLTransE
from VEG.model import Graph_VEG
from scipy import spatial
from veer import VEER

class TestRLTransE(unittest.TestCase):

    def _test_rl_transe(self, model, field_relation_map, params):
        dataset = model()
        graph = Graph_VEG(model)
        logger = get_logger("RL.Test.RLTransE." + str(dataset))
        logger.info("values for name : %s", str(graph.relation_value_map[graph.relation[1]][:10]))
        logger.info("relation: %s", str(graph.relation))
        logger.info("train_triples: %s", str(graph.train_triples[:10]))
        logger.info("set train_triples size %d", len(set(graph.train_triples)))

        transe = RLTransE(graph, dimension=params['dimension'],
                        learning_rate=params['learning_rate'],
                        margin=params['margin'],
                        regularizer_scale=params['regularizer_scale'],
                        batchSize=params['batchSize'],
                        neg_rate=params['neg_rate'],
                        neg_rel_rate=params['neg_rel_rate'])
        loss, val_loss = transe.train(max_epochs=params['epochs'])
        logger.info("Training Complete with loss: %f val_loss: %f", loss, val_loss)

        value_embeddings = transe.get_val_embeddings()
        relation_embeddings = transe.get_rel_embeddings()

        result_prob = []
        distance_distribution = []
        missing_values = []
        for (a, b) in dataset.test_links:
            row_a = dataset.testDataA.loc[a]
            row_b = dataset.testDataB.loc[b]

            distance = 0
            dd = []
            for f in field_relation_map:
                val_a = row_a[f]
                val_b = row_b[f]
                if val_a == val_b:
                    dd.append(0)
                else:
                    rel = field_relation_map[f]
                    try:
                        val_index_a = graph.relation_value_map[rel].index(val_a)
                    except ValueError:
                        missing_values.append(val_a)
                        distance = distance + 1
                        dd.append(1)
                        continue
                    try:
                        val_index_b = graph.relation_value_map[rel].index(val_b)
                    except ValueError:
                        missing_values.append(val_b)
                        distance = distance + 1
                        dd.append(1)
                        continue
                    rel_index = graph.relation.index(field_relation_map[f])

                    cur_distance = abs(spatial.distance.cosine(
                            value_embeddings[rel][val_index_a] + relation_embeddings[rel_index],
                            value_embeddings[rel][val_index_b]))
                    distance = distance + cur_distance
                    dd.append(cur_distance)

            result_prob.append((a, b, distance))
            distance_distribution.append((a, b, dd, distance))
            #logger.info("a: %d, b: %d distance: %f true_pairs: %s", a, b, distance, (a,b) in dataset.true_test_links)
        logger.info("No. of missing values: %d", len(missing_values))
        logger.info("Unique No. of missing values: %d", len(set(missing_values)))
        try:
            entities = ["value\trelation"]
            for r in graph.relation_value_map:
                for v in graph.relation_value_map[r]:
                    entities.append("\t".join([v,r]))

            embeddings = []
            for rel in value_embeddings:
                val_count = len(graph.relation_value_map[rel])
                embeddings.extend(value_embeddings[rel][:val_count])

            #Write Embeddings to file
            export_embeddings('veg', str(dataset), 'RLTransE_val', entities, embeddings)
            export_embeddings('veg', str(dataset), 'RLTransE_rel', graph.relation, relation_embeddings)
        except Exception as e:
            logger.error("Failed to export embeddings")
            logger.error(e)

        optimal_threshold, max_fscore = get_optimal_threshold(result_prob, dataset.true_test_links, max_threshold=3.0, step=0.02)

        try:
            params['threshold'] = optimal_threshold
            result = pd.MultiIndex.from_tuples([(e1, e2) for (e1, e2, d) in result_prob if d <= optimal_threshold])
            log_quality_results(logger, result, dataset.true_test_links, len(dataset.test_links), params)
        except:
            logger.info("Zero Reults")

        #Log MAP, MRR and Hits@K
        ir_metrics = InformationRetrievalMetrics(result_prob, dataset.true_test_links)
        precison_at_1 = ir_metrics.log_metrics(logger, params)

        transe.close_tf_session()

        #Export False Positives and result porobabilities
        get_entity_name = lambda d, i: "_".join([
                                str(d.iloc[i][dataset.field_map[CensusFields.ID_INDIVIDUAL]]),
                                str(d.iloc[i][dataset.field_map[CensusFields.DNI]])])
        get_entity_name_loc = lambda d, i: "_".join([
                                str(d.loc[i][dataset.field_map[CensusFields.ID_INDIVIDUAL]]),
                                str(d.loc[i][dataset.field_map[CensusFields.DNI]])])
        entitiesA = [get_entity_name(dataset.testDataA, i)
                        for i in range(int(dataset.testDataA.shape[0]))]
        entitiesB = [get_entity_name(dataset.testDataB, i)
                        for i in range(int(dataset.testDataB.shape[0]))]
        result_prob = [(entitiesA.index(get_entity_name_loc(dataset.testDataA, int(a))),
                        entitiesB.index(get_entity_name_loc(dataset.testDataB, int(b))),
                        p) for (a, b, p) in result_prob]
        true_links = [(entitiesA.index(get_entity_name_loc(dataset.testDataA, int(a))),
                        entitiesB.index(get_entity_name_loc(dataset.testDataB, int(b))))
                        for (a, b) in dataset.true_test_links]
        export_result_prob(Census, 'veg', 'census', 'rltranse', entitiesA, result_prob,
                                true_links, entitiesB)

        distance_distribution = [(entitiesA.index(get_entity_name_loc(dataset.testDataA, int(a))),
                        entitiesB.index(get_entity_name_loc(dataset.testDataB, int(b))),
                        [str("%.2f" % (float(w))) for w in dd], 1 - d)
                        for (e1, e2, dd, d) in distance_distribution if (e1, e2) in result]
        export_human_readable_results(Census, 'veg', 'census', 'rltranse', entitiesA,
                                        distance_distribution, entitiesB)

        result = [(entitiesA.index(get_entity_name_loc(dataset.testDataA, int(a))),
                    entitiesB.index(get_entity_name_loc(dataset.testDataB, int(b))))
                    for (a, b) in result]
        export_false_negatives(Census, 'veg', 'census', 'rltranse', entitiesA, result_prob,
                                true_links, result, entitiesB)
        export_false_positives(Census, 'veg', 'census', 'rltranse', entitiesA, result_prob,
                                true_links, result, entitiesB)

        return (max_fscore, precison_at_1)

    def get_default_params(self):
        return {'learning_rate': 0.1, 'margin': 1, 'dimension': 128, 'epochs': 1000,
                'regularizer_scale' : 0.1, 'batchSize' : 64, 'neg_rate' : 7, 'neg_rel_rate': 1}

    def test_census(self):
         #Map of feilds in census dataFrame to VEG relations.
        c = Census()
        field_relation_map = {
                    c.field_map[CensusFields.FIRST_NAME] : "name",
                    c.field_map[CensusFields.SURNAME_1] : "surname",
                    c.field_map[CensusFields.YOB] : "yob",
                    c.field_map[CensusFields.CIVIL_STATUS] : "civil",
                    c.field_map[CensusFields.RELATION]: "relation",
                    c.field_map[CensusFields.OCCUPATION] : "occupation"
                }
        return self._test_rl_transe(Census, field_relation_map, self.get_default_params())

    def test_cora(self):
        relation = ['author', 'publisher', 'date', 'title', 'journal', 'volume', 'pages', 'address']
        field_relation_map = {r : r for r in relation}
        return self._test_rl_transe(Cora, field_relation_map, self.get_default_params())

    def test_febrl(self):
        relation = ['given_name', 'surname', 'state', 'date_of_birth', 'postcode']
        field_relation_map = {r: r for r in relation}
        return self._test_rl_transe(FEBRL, field_relation_map, self.get_default_params())

    def test_grid_search_census(self):
         #Map of feilds in census dataFrame to VEG relations.
        c = Census()
        field_relation_map = {
                    c.field_map[CensusFields.FIRST_NAME] : "name",
                    c.field_map[CensusFields.SURNAME_1] : "surname",
                    c.field_map[CensusFields.YOB] : "yob",
                    c.field_map[CensusFields.CIVIL_STATUS] : "civil",
                    c.field_map[CensusFields.RELATION]: "relation",
                    c.field_map[CensusFields.OCCUPATION] : "occupation"
                }

        dimension= [32, 64, 128]
        batchSize= [32, 64, 128]
        learning_rate= [0.1]
        margin= [1, 0.1]
        regularizer_scale = [0.1]
        epochs = [1000, 5000]
        neg_rate = [7]
        neg_rel_rate = [1]

        count = 0
        max_fscore = 0
        max_prec_at_1 = 0

        logger = get_logger('RL.Test.GridSearch.RLTransE.' + str(c))

        for d, bs, lr, m, reg, e, nr, nrr in itertools.product(dimension, batchSize, learning_rate,
                        margin, regularizer_scale, epochs, neg_rate, neg_rel_rate):
            params = {'learning_rate': lr, 'margin': m, 'dimension': d, 'epochs': e, 'batchSize' : bs,
                        'regularizer_scale' : reg, 'neg_rate' : nr, 'neg_rel_rate': nrr}
            logger.info("\nTest:%d, PARAMS: %s", count, str(params))
            count = count + 1
            cur_fscore, cur_prec_at_1 = self._test_rl_transe(Census, field_relation_map, params)
            if max_fscore <= cur_fscore:
                max_fscore = cur_fscore
            if max_prec_at_1 <= cur_prec_at_1:
                max_prec_at_1 = cur_prec_at_1
            logger.info("Ran total %d Tests.", count)
            logger.info("Max Fscore: %f", max_fscore)
            logger.info("Max Mean Precision@1: %f", max_prec_at_1)

        return True

    def test_grid_search_cora(self):
        relation = ['author', 'publisher', 'date', 'title', 'journal', 'volume', 'pages', 'address']
        field_relation_map = {r : r for r in relation}
        c = Cora()

        dimension= [32, 64, 128]
        batchSize= [32, 64, 128]
        learning_rate= [0.1]
        margin= [1, 0.1]
        regularizer_scale = [0.1]
        epochs = [1000, 5000]
        neg_rate = [7]
        neg_rel_rate = [1]

        count = 0
        max_fscore = 0
        max_prec_at_1 = 0

        logger = get_logger('RL.Test.GridSearch.RLTransE.' + str(c))

        for d, bs, lr, m, reg, e, nr, nrr in itertools.product(dimension, batchSize, learning_rate,
                        margin, regularizer_scale, epochs, neg_rate, neg_rel_rate):
            params = {'learning_rate': lr, 'margin': m, 'dimension': d, 'epochs': e, 'batchSize' : bs,
                        'regularizer_scale' : reg, 'neg_rate' : nr, 'neg_rel_rate': nrr}
            logger.info("\nTest:%d, PARAMS: %s", count, str(params))
            count = count + 1
            cur_fscore, cur_prec_at_1 = self._test_rl_transe(Cora, field_relation_map, params)
            if max_fscore <= cur_fscore:
                max_fscore = cur_fscore
            if max_prec_at_1 <= cur_prec_at_1:
                max_prec_at_1 = cur_prec_at_1
            logger.info("Ran total %d Tests.", count)
            logger.info("Max Fscore: %f", max_fscore)
            logger.info("Max Mean Precision@1: %f", max_prec_at_1)

        return True

    def test_grid_search_febrl(self):
        relation = ['given_name', 'surname', 'state', 'date_of_birth', 'postcode']
        field_relation_map = {r: r for r in relation}
        f = FEBRL()

        dimension= [32, 64, 128]
        batchSize= [32, 64, 128]
        learning_rate= [0.1]
        margin= [1, 0.1]
        regularizer_scale = [0.1]
        epochs = [1000, 5000]
        neg_rate = [7]
        neg_rel_rate = [1]

        count = 0
        max_fscore = 0
        max_prec_at_1 = 0

        logger = get_logger('RL.Test.GridSearch.RLTransE.' + str(f))

        for d, bs, lr, m, reg, e, nr, nrr in itertools.product(dimension, batchSize, learning_rate,
                        margin, regularizer_scale, epochs, neg_rate, neg_rel_rate):
            params = {'learning_rate': lr, 'margin': m, 'dimension': d, 'epochs': e, 'batchSize' : bs,
                        'regularizer_scale' : reg, 'neg_rate' : nr, 'neg_rel_rate': nrr}
            logger.info("\nTest:%d, PARAMS: %s", count, str(params))
            count = count + 1
            cur_fscore, cur_prec_at_1 = self._test_rl_transe(FEBRL, field_relation_map, params)
            if max_fscore <= cur_fscore:
                max_fscore = cur_fscore
            if max_prec_at_1 <= cur_prec_at_1:
                max_prec_at_1 = cur_prec_at_1
            logger.info("Ran total %d Tests.", count)
            logger.info("Max Fscore: %f", max_fscore)
            logger.info("Max Mean Precision@1: %f", max_prec_at_1)

        return True