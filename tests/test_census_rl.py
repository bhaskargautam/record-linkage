import unittest
import itertools
import pandas as pd
import numpy as np
import recordlinkage
import unittest

from common import (
    export_embeddings,
    export_result_prob,
    get_optimal_threshold,
    get_logger,
    InformationRetrievalMetrics,
    log_quality_results,
    sigmoid)
from data.base_census import CensusFields
from data.census import Census
from VEG.rltranse import RLTransE
from VEG.model import Graph_VEG
from scipy import spatial


class TestCensusRL(unittest.TestCase):

    def test_rl_transe(self):
        c = Census()
        graph = Graph_VEG(Census)
        logger = get_logger("RL.Test.RLTransE.Census")
        logger.info("values for name : %s", str(graph.relation_value_map[graph.relation[1]][:10]))
        logger.info("relation: %s", str(graph.relation))
        logger.info("train_triples: %s", str(graph.train_triples[:10]))
        logger.info("set train_triples size %d", len(set(graph.train_triples)))

        params = self.get_default_params()
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

        #Map of feilds in census dataFrame to VEG relations.
        field_relation_map = {
                    c.field_map[CensusFields.FIRST_NAME] : "name",
                    c.field_map[CensusFields.SURNAME_1] : "surname",
                    c.field_map[CensusFields.SURNAME_2] : "surname2",
                    c.field_map[CensusFields.YOB] : "yob",
                    c.field_map[CensusFields.CIVIL_STATUS] : "civil",
                    c.field_map[CensusFields.OCCUPATION] : "occupation",
                    c.field_map[CensusFields.RELATION]: "relation"
                }

        result_prob = []
        missing_values = []
        entity_A = []
        entity_A =[]
        for (a, b) in c.test_links:
            row_a = c.testDataA.loc[a]
            row_b = c.testDataB.loc[b]

            distance = 0
            for f in field_relation_map:
                val_a = row_a[f]
                val_b = row_b[f]
                if val_a != val_b:
                    rel = field_relation_map[f]
                    try:
                        val_index_a = graph.relation_value_map[rel].index(val_a)
                    except ValueError:
                        missing_values.append(val_a)
                        continue
                    try:
                        val_index_b = graph.relation_value_map[rel].index(val_b)
                    except ValueError:
                        missing_values.append(val_b)
                        continue
                    rel_index = graph.relation.index(field_relation_map[f])

                    distance = distance + abs(spatial.distance.cosine(
                            value_embeddings[rel][val_index_a] + relation_embeddings[rel_index],
                            value_embeddings[rel][val_index_b]))

            result_prob.append((a, b, distance))
            #logger.info("a: %d, b: %d distance: %f true_pairs: %s", a, b, distance, (a,b) in c.true_test_links)
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
            export_embeddings('veg', str(c), 'RLTransE', entities, embeddings)
        except Exception as e:
            logger.error("Failed to export embeddings")
            logger.error(e)
        #export_result_prob(Census, 'veg', str(c), 'RLTransE', graph.values, result_prob, c.true_test_links)
        optimal_threshold, max_fscore = get_optimal_threshold(result_prob, c.true_test_links, max_threshold=3.0)

        try:
            params['threshold'] = optimal_threshold
            result = pd.MultiIndex.from_tuples([(e1, e2) for (e1, e2, d) in result_prob if d <= optimal_threshold])
            log_quality_results(logger, result, c.true_test_links, len(c.test_links), params)
        except:
            logger.info("Zero Reults")

        #Log MAP, MRR and Hits@K
        ir_metrics = InformationRetrievalMetrics(result_prob, c.true_test_links)
        precison_at_1 = ir_metrics.log_metrics(logger, params)

        transe.close_tf_session()
        return (max_fscore, precison_at_1)

    def get_default_params(self):
        return {'learning_rate': 0.1, 'margin': 2, 'dimension': 128, 'epochs': 1000,
                'regularizer_scale' : 0.1, 'batchSize' : 1024, 'neg_rate' : 8, 'neg_rel_rate': 2}


    def test_ecm(self):
        logger = get_logger('RL.Test.ECMClassifier.Census')

        census = Census()

        compare_cl = census.get_comparision_object()
        features = compare_cl.compute(census.candidate_links, census.trainDataA, census.trainDataB)
        logger.info("Train Features %s", str(features.describe()))

        # Train ECM Classifier
        logrg = recordlinkage.ECMClassifier()
        logrg.fit(features)

        result = logrg.predict(features)
        log_quality_results(logger, result, census.true_links, len(census.candidate_links))

        #Validate the classifier
        compare_cl = census.get_comparision_object()
        features = compare_cl.compute(census.val_links, census.valDataA, census.valDataB)
        logger.info("Validation Features %s", str(features.describe()))

        result = logrg.predict(features)
        log_quality_results(logger, result, census.true_val_links, len(census.val_links))


        #Test the classifier
        compare_cl = census.get_comparision_object()
        features = compare_cl.compute(census.test_links, census.testDataA, census.testDataB)
        logger.info("Test Features %s", str(features.describe()))

        result = logrg.predict(features)
        log_quality_results(logger, result, census.true_test_links, len(census.test_links))

        #Log IR Stats: MRR, MAP, MP@K
        prob_series = logrg.prob(features)
        prob = [(1 - p) for p in prob_series.tolist()]
        result_prob = [(census.test_links[i][0], census.test_links[i][1], prob[i]) for i in range(0, len(prob))]
        ir_metrics = InformationRetrievalMetrics(result_prob, census.true_test_links)
        ir_metrics.log_metrics(logger)

    def test_logistic(self):
        logger = get_logger('RL.Test.LogisticRegression.Census')

        census = Census()

        compare_cl = census.get_comparision_object()
        features = compare_cl.compute(census.candidate_links, census.trainDataA, census.trainDataB)
        logger.info("Train Features %s", str(features.describe()))

        # Train ECM Classifier
        logrg = recordlinkage.LogisticRegressionClassifier()
        logrg.fit(features, census.true_links)

        result = logrg.predict(features)
        log_quality_results(logger, result, census.true_links, len(census.candidate_links))

        #Validate the classifier
        compare_cl = census.get_comparision_object()
        features = compare_cl.compute(census.val_links, census.valDataA, census.valDataB)
        logger.info("Validation Features %s", str(features.describe()))
        result = logrg.predict(features)
        log_quality_results(logger, result, census.true_val_links, len(census.val_links))

        #Test the classifier
        compare_cl = census.get_comparision_object()
        features = compare_cl.compute(census.test_links, census.testDataA, census.testDataB)
        logger.info("Test Features %s", str(features.describe()))

        result = logrg.predict(features)
        log_quality_results(logger, result, census.true_test_links, len(census.test_links))

        #Log IR Stats: MRR, MAP, MP@K
        prob_series = logrg.prob(features)
        prob = [(1 - p) for p in prob_series.tolist()]
        result_prob = [(census.test_links[i][0], census.test_links[i][1], prob[i]) for i in range(0, len(prob))]
        ir_metrics = InformationRetrievalMetrics(result_prob, census.true_test_links)
        ir_metrics.log_metrics(logger)