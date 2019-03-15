import config
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
from data.cora import Cora
from data.febrl import FEBRL
from data.census import Census
from ER.model import Graph_ER
from ER.transe import TransE
from scipy import spatial
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import precision_recall_curve


#def get_distance(entA, entB, ent_embeddings):

class TestTransE(unittest.TestCase):
    def test_cora(self, params=None):
        if not params:
            params = self.get_default_params()

        #Load Graph Data
        graph = Graph_ER(Cora)
        model = Cora()
        logger = get_logger('RL.Test.TransE.Household.' + str(model))

        transe = TransE(graph, dimension=params['dimension'],
                        learning_rate=params['learning_rate'],
                        margin=params['margin'],
                        regularizer_scale=params['regularizer_scale'],
                        batchSize=params['batchSize'],
                        neg_rate=params['neg_rate'],
                        neg_rel_rate=params['neg_rel_rate'])
        loss = transe.train(max_epochs=params['epochs'])
        logger.info("Training Complete with loss: %f", loss)

        ent_embeddings = transe.get_ent_embeddings()

        #Experimenting household matching
        auth_rel_index = graph.relation.index('author')
        result_prob = []
        for ep_index in range(0, len(graph.entity_pairs)):
            authors_A = [t for (h,t,r) in graph.triples if h == graph.entity_pairs[ep_index][0] and r == auth_rel_index]
            #logger.info("AUHTORS A: %s", str([graph.entity[a] for a in authors_A]))
            authors_B = [t for (h,t,r) in graph.triples if h == graph.entity_pairs[ep_index][1] and r == auth_rel_index]
            #logger.info("AUHTORS B: %s", str([graph.entity[a] for a in authors_B]))

            cost_matrix = np.zeros(shape=(len(authors_A), len(authors_B)))
            for i in range(len(authors_A)):
                for j in range(len(authors_B)):
                    #if authors_A[i] == authors_B[j]:
                    #    cost_matrix[i][j] = 100
                    #else:
                    cost_matrix[i][j] = abs(spatial.distance.cosine(
                            ent_embeddings[authors_A[i]],
                            ent_embeddings[authors_B[j]]))

            #logger.info("Cost Matrix: %s", str(cost_matrix))

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            #logger.info("Cost of aligning = %f", cost_matrix[row_ind, col_ind].sum())
            distance = cost_matrix[row_ind, col_ind].sum() + abs(spatial.distance.cosine(
                                ent_embeddings[graph.entity_pairs[ep_index][0]],
                                ent_embeddings[graph.entity_pairs[ep_index][1]]))
            result_prob.append((graph.entity_pairs[ep_index][0], graph.entity_pairs[ep_index][1], distance))
            if distance <= 0.05:
                logger.info("i: %d, distance: %f true_pairs: %s", ep_index, distance,
                        graph.entity_pairs[ep_index] in graph.true_pairs)

        export_embeddings('er', str(model), 'TransE.Household', graph.entity, ent_embeddings)
        export_result_prob(Cora, 'er', str(model), 'TransE.Household', graph.entity, result_prob, graph.true_pairs)
        optimal_threshold, max_fscore = get_optimal_threshold(result_prob, graph.true_pairs)
        try:
            params['threshold'] = optimal_threshold
            result = pd.MultiIndex.from_tuples([(e1, e2) for (e1, e2, d) in result_prob if d <= optimal_threshold])
            log_quality_results(logger, result, graph.true_pairs, len(graph.entity_pairs), params)
        except:
            logger.info("Zero Reults")

        #Log MAP, MRR and Hits@K
        ir_metrics = InformationRetrievalMetrics(result_prob, graph.true_pairs)
        ir_metrics.log_metrics(logger, params)

        transe.close_tf_session()
        return max_fscore

    def test_census(self, params=None):
        if not params:
            params = self.get_default_params()

        #Load Graph Data
        graph = Graph_ER(Census)
        model = Census()
        logger = get_logger('RL.Test.TransE.Household.' + str(model))

        transe = TransE(graph, dimension=params['dimension'],
                        learning_rate=params['learning_rate'],
                        margin=params['margin'],
                        regularizer_scale=params['regularizer_scale'],
                        batchSize=params['batchSize'],
                        neg_rate=params['neg_rate'],
                        neg_rel_rate=params['neg_rel_rate'])
        loss = transe.train(max_epochs=params['epochs'])
        logger.info("Training Complete with loss: %f", loss)

        ent_embeddings = transe.get_ent_embeddings()

        #Experimenting household matching
        result_prob = []
        for ep_index in range(0, len(graph.entity_pairs)):
            #logger.info("Computing cost for: %s", str([graph.entity[e] for e in graph.entity_pairs[ep_index]]))
            household_A = [t for (h,t,r) in graph.triples if h == graph.entity_pairs[ep_index][0] and r > 6][0]
            family_members_A = [h for (h,t,r) in graph.triples if t == household_A]
            #logger.info("FM A: %s", str([graph.entity[a] for a in family_members_A]))
            household_B = [t for (h,t,r) in graph.triples if h == graph.entity_pairs[ep_index][1] and r > 6][0]
            family_members_B = [h for (h,t,r) in graph.triples if t == household_B]
            #logger.info("FM B: %s", str([graph.entity[a] for a in family_members_B]))

            cost_matrix = np.zeros(shape=(len(family_members_A), len(family_members_B)))
            for i in range(len(family_members_A)):
                for j in range(len(family_members_B)):
                    #if family_members_A[i] == family_members_B[j]:
                    #    cost_matrix[i][j] = 100
                    #else:
                    cost_matrix[i][j] = abs(spatial.distance.cosine(
                            ent_embeddings[family_members_A[i]],
                            ent_embeddings[family_members_B[j]]))

            #logger.info("Cost Matrix: %s", str(cost_matrix))

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            #logger.info("Cost of aligning = %f", cost_matrix[row_ind, col_ind].sum())
            #logger.info("Rows selected %s, Col selected: %s", str(row_ind), str(col_ind))

            eA_index = family_members_A.index(graph.entity_pairs[ep_index][0])
            eB_index = family_members_B.index(graph.entity_pairs[ep_index][1])
            #logger.info("A index: %d, B index: %d", eA_index, eB_index)

            rowA = np.where(row_ind == eA_index)[0]
            if len(rowA) and col_ind[rowA[0]] == eB_index:
                #logger.info("Pair in min. cost matrix")
                distance = cost_matrix[row_ind, col_ind].sum()
            else:
                distance = cost_matrix[row_ind, col_ind].sum() + abs(spatial.distance.cosine(
                                ent_embeddings[graph.entity_pairs[ep_index][0]],
                                ent_embeddings[graph.entity_pairs[ep_index][1]]))

            result_prob.append((graph.entity_pairs[i][0], graph.entity_pairs[i][1], distance))
            if ep_index % 1000 == 0:
                logger.info("i: %d, distance: %f true_pairs: %s", ep_index, distance,
                        graph.entity_pairs[ep_index] in graph.true_pairs)
            #if graph.entity_pairs[ep_index] in graph.true_pairs:
            #    import ipdb;ipdb.set_trace()
        #Normalize distance
        max_distance = 10
        #for r in result_prob:
        #    if r[2] > max_distance:
        #        max_distance = r[2]
        result_prob = [(r[0], r[1], (r[2]/max_distance)) for r in result_prob]
        #logger.info("Max distance: %f", max_distance)

        for r in result_prob[:100]:
            logger.info("distance: %f true_pairs: %s", r[2], (r[0], r[1]) in graph.true_pairs)
        export_embeddings('er', str(model), 'TransE.Household', graph.entity, ent_embeddings)
        export_result_prob(Census, 'er', str(model), 'TransE.Household', graph.entity, result_prob, graph.true_pairs)
        optimal_threshold, max_fscore = get_optimal_threshold(result_prob, graph.true_pairs)
        try:
            params['threshold'] = optimal_threshold
            result = pd.MultiIndex.from_tuples([(e1, e2) for (e1, e2, d) in result_prob if d <= optimal_threshold])
            log_quality_results(logger, result, graph.true_pairs, len(graph.entity_pairs), params)
        except:
            logger.info("Zero Reults")

        #Log MAP, MRR and Hits@K
        ir_metrics = InformationRetrievalMetrics(result_prob, graph.true_pairs)
        ir_metrics.log_metrics(logger, params)

        transe.close_tf_session()
        return max_fscore

    def test_febrl(self, params=None):
        if not params:
            params = self.get_default_params()

        #Load Graph Data
        graph = Graph_ER(FEBRL)
        model = FEBRL()
        logger = get_logger('RL.Test.TransE.Household.' + str(model))

        transe = TransE(graph, dimension=params['dimension'],
                        learning_rate=params['learning_rate'],
                        margin=params['margin'],
                        regularizer_scale=params['regularizer_scale'],
                        batchSize=params['batchSize'],
                        neg_rate=params['neg_rate'],
                        neg_rel_rate=params['neg_rel_rate'])
        loss = transe.train(max_epochs=params['epochs'])
        logger.info("Training Complete with loss: %f", loss)

        ent_embeddings = transe.get_ent_embeddings()

        #Experimenting household matching
        postcode_rel_id = graph.relation.index("postcode")
        result_prob = []
        for i in range(0, len(graph.entity_pairs)):
            person_A = graph.entity_pairs[i][0]
            person_B = graph.entity_pairs[i][1]

            postcode_A = [t for (h,t,r) in graph.triples if h == person_A and r == postcode_rel_id][0]
            neighbours_A = [h for (h,t,r) in graph.triples if t == postcode_A]
            #logger.info("FM A: %s", str([graph.entity[a] for a in neighbours_A]))
            postcode_B = [t for (h,t,r) in graph.triples if h == person_B and r == postcode_rel_id][0]
            neighbours_B = [h for (h,t,r) in graph.triples if t == postcode_B]
            #logger.info("FM B: %s", str([graph.entity[a] for a in neighbours_B]))

            cost_matrix = np.zeros(shape=(len(neighbours_A), len(neighbours_B)))
            for i in range(len(neighbours_A)):
                for j in range(len(neighbours_B)):
                    if neighbours_A[i] == neighbours_B[j]:
                        cost_matrix[i][j] = 100
                    else:
                        cost_matrix[i][j] = abs(spatial.distance.cosine(
                                ent_embeddings[neighbours_A[i]],
                                ent_embeddings[neighbours_B[j]]))


            #logger.info("Cost Matrix: %s", str(cost_matrix))

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            #logger.info("Cost of aligning = %f", cost_matrix[row_ind, col_ind].sum())

            person_A_index = neighbours_A.index(person_A)
            person_B_index = neighbours_B.index(person_B)
            distance = cost_matrix[row_ind, col_ind].sum() + cost_matrix[person_A_index][person_B_index]
            #import ipdb;ipdb.set_trace()
            #if (person_A_index, person_B_index) not in (row_ind, col_ind):
            #   distance = distance + 1000

            result_prob.append((graph.entity_pairs[i][0], graph.entity_pairs[i][1], distance))

        export_embeddings('er', str(model), 'TransE.Household', graph.entity, ent_embeddings)
        export_result_prob(FEBRL, 'er', str(model), 'TransE.Household', graph.entity, result_prob, graph.true_pairs)
        optimal_threshold, max_fscore = get_optimal_threshold(result_prob, graph.true_pairs)
        try:
            params['threshold'] = optimal_threshold
            result = pd.MultiIndex.from_tuples([(e1, e2) for (e1, e2, d) in result_prob if d <= optimal_threshold])
            log_quality_results(logger, result, graph.true_pairs, len(graph.entity_pairs), params)
        except:
            logger.info("Zero Reults")

        #Log MAP, MRR and Hits@K
        ir_metrics = InformationRetrievalMetrics(result_prob, graph.true_pairs)
        ir_metrics.log_metrics(logger, params)

        transe.close_tf_session()
        return max_fscore

    def get_default_params(self):
        return {'learning_rate': 0.1, 'margin': 1, 'dimension': 80, 'epochs': 100,
                'regularizer_scale' : 0.1, 'batchSize' : 100, 'neg_rate' : 8, 'neg_rel_rate': 2}

    def test_transe_cora(self):
        self._test_transe(Cora, self.get_default_params())

    def test_transe_febrl(self):
        self._test_transe(FEBRL, self.get_default_params())

    def test_transe_census(self):
        self._test_transe(Census, self.get_default_params())

    def _test_grid_search(self, dataset):
        dimension= [50, 80, 120]
        batchSize= [100]
        learning_rate= [0.1, 0.2]
        margin= [0.5, 1]
        regularizer_scale = [0.1, 0.2]
        epochs = [100, 500]
        neg_rel_rate = [1, 2, 5]
        neg_rate = [1, 5, 10]
        count = 0
        max_fscore = 0

        model = dataset()
        logger = get_logger('RL.Test.GridSearch.TransE.' + str(model))

        for d, bs, lr, m, reg, e, nr, nrr in \
            itertools.product(dimension, batchSize, learning_rate, margin, regularizer_scale, epochs, neg_rate, neg_rel_rate):
            params = {'learning_rate': lr, 'margin': m, 'dimension': d, 'epochs': e, 'batchSize' : bs,
                        'regularizer_scale' : reg, 'neg_rate' : nr, 'neg_rel_rate': nrr}
            logger.info("\nPARAMS: %s", str(params))
            count = count + 1
            cur_fscore = self._test_transe(dataset, params)
            if max_fscore <= cur_fscore:
                max_fscore = cur_fscore

        logger.info("Ran total %d Tests.", count)
        logger.info("Max Fscore: %f", max_fscore)

    def test_grid_search_cora(self):
        self._test_grid_search(Cora)

    def test_grid_search_febrl(self):
        self._test_grid_search(FEBRL)

    def test_grid_search_census(self):
        self._test_grid_search(Census)