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
from VEG.rltranse import RLTransE
from VEG.model import Graph_VEG
from scipy import spatial
from veer import VEER

class TestCensusRL(unittest.TestCase):

    def test_rl_transe(self):
        c = Census()
        graph = Graph_VEG(Census)
        logger = get_logger("RL.Test.RLTransE.Census")
        logger.info("values for name : %s", str(graph.relation_value_map[graph.relation[1]][:10]))
        logger.info("relation: %s", str(graph.relation))
        logger.info("train_triples: %s", str(graph.train_triples[:10]))
        logger.info("set train_triples size %d", len(set(graph.train_triples)))

        params = {'learning_rate': 0.1, 'margin': 1, 'dimension': 256, 'epochs': 1000,
                'regularizer_scale' : 0.1, 'batchSize' : 128, 'neg_rate' : 8, 'neg_rel_rate': 2}
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
                    c.field_map[CensusFields.YOB] : "yob",
                    c.field_map[CensusFields.CIVIL_STATUS] : "civil",
                    c.field_map[CensusFields.RELATION]: "relation",
                    c.field_map[CensusFields.OCCUPATION] : "occupation"
                }

        result_prob = []
        distance_distribution = []
        missing_values = []
        for (a, b) in c.test_links:
            row_a = c.testDataA.loc[a]
            row_b = c.testDataB.loc[b]

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
            export_embeddings('veg', str(c), 'RLTransE_val', entities, embeddings)
            export_embeddings('veg', str(c), 'RLTransE_rel', graph.relation, relation_embeddings)
        except Exception as e:
            logger.error("Failed to export embeddings")
            logger.error(e)

        optimal_threshold, max_fscore = get_optimal_threshold(result_prob, c.true_test_links, max_threshold=3.0, step=0.02)

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

        #Export False Positives and result porobabilities
        get_entity_name = lambda c, d, i: "_".join([
                                str(d.iloc[i][c.field_map[CensusFields.ID_INDIVIDUAL]]),
                                str(d.iloc[i][c.field_map[CensusFields.DNI]])])
        get_entity_name_loc = lambda c, d, i: "_".join([
                                str(d.loc[i][c.field_map[CensusFields.ID_INDIVIDUAL]]),
                                str(d.loc[i][c.field_map[CensusFields.DNI]])])
        entitiesA = [get_entity_name(c, c.testDataA, i)
                        for i in range(int(c.testDataA.shape[0]))]
        entitiesB = [get_entity_name(c, c.testDataB, i)
                        for i in range(int(c.testDataB.shape[0]))]
        result_prob = [(entitiesA.index(get_entity_name_loc(c, c.testDataA, int(a))),
                        entitiesB.index(get_entity_name_loc(c, c.testDataB, int(b))),
                        p) for (a, b, p) in result_prob]
        true_links = [(entitiesA.index(get_entity_name_loc(c, c.testDataA, int(a))),
                        entitiesB.index(get_entity_name_loc(c, c.testDataB, int(b))))
                        for (a, b) in c.true_test_links]
        export_result_prob(Census, 'veg', 'census', 'rltranse', entitiesA, result_prob,
                                true_links, entitiesB)

        distance_distribution = [(entitiesA.index(get_entity_name_loc(c, c.testDataA, int(a))),
                        entitiesB.index(get_entity_name_loc(c, c.testDataB, int(b))),
                        [str("%.2f" % (float(w))) for w in dd], 1 - d)
                        for (e1, e2, dd, d) in distance_distribution if (e1, e2) in result]
        export_human_readable_results(Census, 'veg', 'census', 'rltranse', entitiesA,
                                        distance_distribution, entitiesB)

        result = [(entitiesA.index(get_entity_name_loc(c, c.testDataA, int(a))),
                    entitiesB.index(get_entity_name_loc(c, c.testDataB, int(b))))
                    for (a, b) in result]
        export_false_negatives(Census, 'veg', 'census', 'rltranse', entitiesA, result_prob,
                                true_links, result, entitiesB)
        export_false_positives(Census, 'veg', 'census', 'rltranse', entitiesA, result_prob,
                                true_links, result, entitiesB)

        return (max_fscore, precison_at_1)

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

        logger.info("ECM weights: %s", str(logrg.weights))

        #Log IR Stats: MRR, MAP, MP@K
        prob_series = logrg.prob(features)
        prob = [(1 - p) for p in prob_series.tolist()]
        result_prob = [(census.test_links[i][0], census.test_links[i][1], prob[i]) for i in range(0, len(prob))]
        ir_metrics = InformationRetrievalMetrics(result_prob, census.true_test_links)
        ir_metrics.log_metrics(logger)


        #Export False Positives and result porobabilities
        result_feature_mapping = [(e1, e2, [str(v) for v in features.loc[(e1, e2)].values], d)
                            for (e1, e2, d) in result_prob if (e1, e2) in result]

        get_entity_name = lambda c, d, i: "_".join([
                                str(d.iloc[i][c.field_map[CensusFields.ID_INDIVIDUAL]]),
                                str(d.iloc[i][c.field_map[CensusFields.DNI]])])
        get_entity_name_loc = lambda c, d, i: "_".join([
                                str(d.loc[i][c.field_map[CensusFields.ID_INDIVIDUAL]]),
                                str(d.loc[i][c.field_map[CensusFields.DNI]])])
        start_time = timeit.default_timer()
        entitiesA = [get_entity_name(census, census.testDataA, i)
                        for i in range(int(census.testDataA.shape[0]))]
        entitiesB = [get_entity_name(census, census.testDataB, i)
                        for i in range(int(census.testDataB.shape[0]))]
        logger.info("Entities built in %s", str(timeit.default_timer() - start_time))

        start_time = timeit.default_timer()
        result_prob = [(entitiesA.index(get_entity_name_loc(census, census.testDataA, int(a))),
                        entitiesB.index(get_entity_name_loc(census, census.testDataB, int(b))),
                        p) for (a, b, p) in result_prob]
        logger.info("Result prob in %s", str(timeit.default_timer() - start_time))

        start_time = timeit.default_timer()
        true_links = [(entitiesA.index(get_entity_name_loc(census, census.testDataA, int(a))),
                        entitiesB.index(get_entity_name_loc(census, census.testDataB, int(b))))
                        for (a, b) in census.true_test_links]
        logger.info("true_links in %s", str(timeit.default_timer() - start_time))

        start_time = timeit.default_timer()
        export_result_prob(Census, 'ECM', 'census', 'ecm', entitiesA, result_prob,
                                true_links, entitiesB)
        logger.info("Result prob EXPORTED in %s", str(timeit.default_timer() - start_time))

        start_time = timeit.default_timer()
        result = [(entitiesA.index(get_entity_name_loc(census, census.testDataA, int(a))),
                    entitiesB.index(get_entity_name_loc(census, census.testDataB, int(b))))
                    for (a, b) in result]
        export_false_negatives(Census, 'ECM', 'census', 'ecm', entitiesA, result_prob,
                                true_links, result, entitiesB)
        export_false_positives(Census, 'ECM', 'census', 'ecm', entitiesA, result_prob,
                                true_links, result, entitiesB)
        logger.info("FP & FN EXPORTED in %s", str(timeit.default_timer() - start_time))


        result_feature_mapping = [(entitiesA.index(get_entity_name_loc(census, census.testDataA, int(a))),
                        entitiesB.index(get_entity_name_loc(census, census.testDataB, int(b))),
                        w, p) for (a, b, w, p) in result_feature_mapping]
        export_human_readable_results(Census, 'ECM', 'census', 'ecm', entitiesA,
                                        result_feature_mapping, entitiesB)
        logger.info("Exported Human Readable Results")


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

        logger.info("logrg coefficients: %s", str(logrg.coefficients))
        #Log IR Stats: MRR, MAP, MP@K
        prob_series = logrg.prob(features)
        prob = [(1 - p) for p in prob_series.tolist()]
        result_prob = [(census.test_links[i][0], census.test_links[i][1], prob[i]) for i in range(0, len(prob))]
        ir_metrics = InformationRetrievalMetrics(result_prob, census.true_test_links)
        ir_metrics.log_metrics(logger)

        #Export False Positives and result porobabilities
        result_feature_mapping = [(e1, e2, [str(v) for v in features.loc[(e1, e2)].values], d)
                            for (e1, e2, d) in result_prob if (e1, e2) in result]

        get_entity_name = lambda c, d, i: "_".join([
                                str(d.iloc[i][c.field_map[CensusFields.ID_INDIVIDUAL]]),
                                str(d.iloc[i][c.field_map[CensusFields.DNI]])])
        get_entity_name_loc = lambda c, d, i: "_".join([
                                str(d.loc[i][c.field_map[CensusFields.ID_INDIVIDUAL]]),
                                str(d.loc[i][c.field_map[CensusFields.DNI]])])
        entitiesA = [get_entity_name(census, census.testDataA, i)
                        for i in range(int(census.testDataA.shape[0]))]
        entitiesB = [get_entity_name(census, census.testDataB, i)
                        for i in range(int(census.testDataB.shape[0]))]
        result_prob = [(entitiesA.index(get_entity_name_loc(census, census.testDataA, int(a))),
                        entitiesB.index(get_entity_name_loc(census, census.testDataB, int(b))),
                        p) for (a, b, p) in result_prob]
        true_links = [(entitiesA.index(get_entity_name_loc(census, census.testDataA, int(a))),
                        entitiesB.index(get_entity_name_loc(census, census.testDataB, int(b))))
                        for (a, b) in census.true_test_links]
        export_result_prob(Census, 'LogisticRegression', 'census', 'logistic', entitiesA,
                                result_prob, true_links, entitiesB)

        result = [(entitiesA.index(get_entity_name_loc(census, census.testDataA, int(a))),
                    entitiesB.index(get_entity_name_loc(census, census.testDataB, int(b))))
                    for (a, b) in result]
        export_false_negatives(Census, 'LogisticRegression', 'census', 'logistic', entitiesA,
                        result_prob, true_links, result, entitiesB)
        export_false_positives(Census, 'LogisticRegression', 'census', 'logistic', entitiesA,
                        result_prob, true_links, result, entitiesB)

        weights = logrg.coefficients
        result = [(e1, e2, [str("%.2f" % (float(d * w)/sum(weights))) for w in weights], d)
                for (e1, e2, d) in result_prob if (e1, e2) in result]

        result_feature_mapping = [(
                        entitiesA.index(get_entity_name_loc(census, census.testDataA, int(a))),
                        entitiesB.index(get_entity_name_loc(census, census.testDataB, int(b))),
                        w, p) for (a, b, w, p) in result_feature_mapping]
        export_human_readable_results(Census, 'LogisticRegression', 'census', 'logistic',
                                        entitiesA, result_feature_mapping, entitiesB)

    def test_veer(self):
        logger = get_logger('RL.Test.VEER.Census')

        dataset = Census()

        #Columns of interest for Sant Feliu town
        columns = ['Noms_harmo', 'cognom_1', 'cohort', 'estat_civil',
                    'parentesc_har', 'ocupacio_hisco']
        params = {'learning_rate': 0.1, 'margin': 0.1, 'dimension': 32, 'epochs': 50,
                    'regularizer_scale' : 0.1, 'batchSize' : 512}

        veer = VEER(Census, columns, dimension=params['dimension'],
                        learning_rate=params['learning_rate'],
                        margin=params['margin'],
                        regularizer_scale=params['regularizer_scale'],
                        batchSize=params['batchSize'])

        #Train Model
        loss, val_loss = veer.train(max_epochs=params['epochs'])
        logger.info("Training Complete with loss: %f, val_loss:%f", loss, val_loss)

        #Test Model
        result_prob, accuracy = veer.test()
        logger.info("Predict count: %d", len(result_prob))
        logger.info("Sample Prob: %s", str([ (c, (a, b) in dataset.true_test_links)
                                        for (a,b,c) in result_prob[:20]]))
        logger.info("Column Weights: %s", str(veer.get_col_weights()))
        logger.info("Accuracy: %s", str(accuracy))
        logger.info("Sample embeddings: %s", str(veer.get_val_embeddings()[0]))

        #Compute Performance measures
        optimal_threshold, max_fscore = get_optimal_threshold(result_prob, dataset.true_test_links, max_threshold=2.0)

        try:
            params['threshold'] = optimal_threshold
            result = pd.MultiIndex.from_tuples([(e1, e2) for (e1, e2, d) in result_prob if d <= optimal_threshold])
            log_quality_results(logger, result, dataset.true_test_links, len(dataset.test_links), params)
        except Exception as e:
            logger.info("Zero Reults")
            logger.error(e)

        #Log MAP, MRR and Hits@K
        ir_metrics = InformationRetrievalMetrics(result_prob, dataset.true_test_links)
        precison_at_1 = ir_metrics.log_metrics(logger, params)

        #Export embeddings
        embeddings = veer.get_val_embeddings()
        export_embeddings('veg', 'census', 'veer', veer.values, embeddings)

        #Write Result Prob to file
        result_feature_mapping = [(e1, e2, [str(abs(spatial.distance.cosine(
                    embeddings[veer.values.index(veer._clean(dataset.testDataA.loc[e1][c]))],
                    embeddings[veer.values.index(veer._clean(dataset.testDataB.loc[e2][c]))])))
                    for c in columns], d)
                    for (e1, e2, d) in result_prob if (e1, e2) in result]

        entitiesA = dataset.get_entity_names(dataset.testDataA)
        entitiesB = dataset.get_entity_names(dataset.testDataB)
        index_dictA = {str(dataset.testDataA.iloc[i]._name) : i
                        for i in range(dataset.testDataA.shape[0])}
        index_dictB = {str(dataset.testDataB.iloc[i]._name) : i
                        for i in range(dataset.testDataB.shape[0])}
        result_prob = [(index_dictA[str(a)], index_dictB[str(b)], p)
                            for (a, b, p) in result_prob]
        export_result_prob(dataset, 'veg', str(dataset), 'VEER', entitiesA, result_prob,
                                    dataset.true_test_links, entitiesB)
        export_false_negatives(Census, 'veg', str(dataset), 'VEER', entitiesA, result_prob,
                            dataset.true_test_links, result, entitiesB)
        export_false_positives(Census, 'veg', str(dataset), 'VEER', entitiesA, result_prob,
                            dataset.true_test_links, result, entitiesB)

        result_feature_mapping = [(index_dictA[str(a)], index_dictB[str(b)],
                        w, p) for (a, b, w, p) in result_feature_mapping]
        export_human_readable_results(Census, 'veg', str(dataset), 'VEER', entitiesA,
                                        result_feature_mapping, entitiesB)

        veer.close_tf_session()

