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


class TestLogisticRLTransE(unittest.TestCase):

    def test_census_new(self):
        c = Census()
        graph = Graph_VEG(Census)
        logger = get_logger("RL.Test.LogisticRLTransE.Census")
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

        missing_values = []
        train_features = [] #Size samples*(dimension*rel_count)
        test_features = []
        for (candidate_links, dataA, dataB, features) in \
                            [(c.candidate_links, c.trainDataA, c.trainDataB, train_features),
                            (c.test_links, c.testDataA, c.testDataB, test_features)]:
            for (a, b) in candidate_links:
                row_a = dataA.loc[a]
                row_b = dataB.loc[b]
                distance = []

                for f in field_relation_map:
                    val_a = row_a[f]
                    val_b = row_b[f]
                    if val_a != val_b:
                        rel = field_relation_map[f]
                        try:
                            val_index_a = graph.relation_value_map[rel].index(val_a)
                        except ValueError:
                            missing_values.append(val_a)
                            distance.extend([1.0] * params['dimension'])
                            continue
                        try:
                            val_index_b = graph.relation_value_map[rel].index(val_b)
                        except ValueError:
                            missing_values.append(val_b)
                            distance.extend([1.0] * params['dimension'])
                            continue
                        rel_index = graph.relation.index(field_relation_map[f])

                        distance.extend(value_embeddings[rel][val_index_a] + \
                            relation_embeddings[rel_index] - value_embeddings[rel][val_index_b])

                features.append(pd.Series(distance).rename((a, b)))
                #logger.info("a: %d, b: %d distance: %f true_pairs: %s", a, b, distance, (a,b) in c.true_test_links)
        logger.info("No. of missing values: %d", len(missing_values))
        logger.info("Unique No. of missing values: %d", len(set(missing_values)))


        train_features = pd.DataFrame(data=train_features).fillna(1)
        test_features = pd.DataFrame(data=test_features).fillna(1)
        logger.info("Shape of Train features: %s", str(train_features.shape))
        logger.info("Shape of Test features: %s", str(test_features.shape))

        #Train Logistic Regression Model
        logrg = recordlinkage.LogisticRegressionClassifier()
        logrg.fit(train_features, c.true_links)
        result = logrg.predict(train_features)
        result = pd.MultiIndex.from_tuples(result.to_series())
        log_quality_results(logger, result, c.true_links, len(c.candidate_links), params)

        #Test Classifier
        result = logrg.predict(test_features)
        result = pd.MultiIndex.from_tuples(result.to_series())
        log_quality_results(logger, result, c.true_test_links, len(c.test_links), params)

        """
        Todo: Export Embeddings and probabilities.
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
            export_embeddings('veg', str(c), 'LogisticRLTransE', entities, embeddings)
        except Exception as e:
            logger.error("Failed to export embeddings")
            logger.error(e)
        export_result_prob(Census, 'veg', str(c), 'RLTransE', graph.values, result_prob, c.true_test_links)
        """
        prob_series = logrg.prob(test_features)
        prob = [(1 - p) for p in prob_series.tolist()]
        result_prob = [(c.test_links[i][0], c.test_links[i][1], prob[i]) for i in range(0, len(prob))]
        #Log MAP, MRR and Hits@K
        ir_metrics = InformationRetrievalMetrics(result_prob, c.true_test_links)
        precison_at_1 = ir_metrics.log_metrics(logger, params)

        transe.close_tf_session()

    def get_default_params(self):
        return {'learning_rate': 0.1, 'margin': 2, 'dimension': 32, 'epochs': 10,
                'regularizer_scale' : 0.1, 'batchSize' : 128, 'neg_rate' : 8, 'neg_rel_rate': 2}
