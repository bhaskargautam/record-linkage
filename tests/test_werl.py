import config
import itertools
import pandas as pd
import numpy as np
import recordlinkage
import unittest

from common import (
    export_embeddings,
    export_false_positives,
    export_false_negatives,
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
from ER.transh import TransH
from VEG.model import Graph_VEG
from VEG.rltranse import RLTransE
from werl import WERL
from veer import VEER
from scipy import spatial

class TestWERL(unittest.TestCase):

    def _get_tf_model_filename(self, dataset, ea_method):
        return 'out/werl/' + str(dataset) + "_" + str(ea_method)

    def _test_werl(self, model, columns, params):
        #Load Graph Data
        dataset = model()
        logger = get_logger('RL.Test.WERL.' + str(dataset))
        ea_params = self.get_optimal_ea_params(model, params['ea_method'])
        if params['ea_method'] in [TransE, TransH]:
            #ER methods
            graph = Graph_ER(model)
            #Train TransE embedding vectors
            transe = params['ea_method'](graph,
                            dimension=ea_params['dimension'],
                            learning_rate=ea_params['learning_rate'],
                            margin=ea_params['margin'],
                            regularizer_scale=ea_params['regularizer_scale'],
                            batchSize=ea_params['batchSize'],
                            neg_rate=ea_params['neg_rate'],
                            neg_rel_rate=ea_params['neg_rel_rate'])
            try:
                #raise Exception("Reset")
                transe.restore_model(self._get_tf_model_filename(dataset, transe))
            except Exception as e:
                logger.error(e)
                loss = transe.train(max_epochs=ea_params['epochs'])
                logger.info("Training Complete with loss: %f", loss)
                transe.save_model(self._get_tf_model_filename(dataset, transe))

            ent_embeddings = transe.get_ent_embeddings()
            rel_embeddings = None
            entity = graph.entity
            transe.close_tf_session()
        elif params['ea_method'] in [RLTransE]:
            #VEG methods
            graph = Graph_VEG(model)
            #Train TransE embedding vectors
            rltranse = params['ea_method'](graph,
                            dimension=ea_params['dimension'],
                            learning_rate=ea_params['learning_rate'],
                            margin=ea_params['margin'],
                            regularizer_scale=ea_params['regularizer_scale'],
                            batchSize=ea_params['batchSize'],
                            neg_rate=ea_params['neg_rate'],
                            neg_rel_rate=ea_params['neg_rel_rate'])

            try:
                #raise Exception("Reset")
                rltranse.restore_model(self._get_tf_model_filename(dataset, rltranse))
            except Exception as e:
                logger.error(e)
                loss, val_loss = rltranse.train(max_epochs=ea_params['epochs'])
                logger.info("Training Complete with loss: %f", loss)
                rltranse.save_model(self._get_tf_model_filename(dataset, rltranse))

            val_embeddings = rltranse.get_val_embeddings()
            rel_embeddings = rltranse.get_rel_embeddings()
            if model == Census:
                #hack: census veg graph has 8 relations. we need only 6
                #removing same_as and surname2 embedding.
                rel_embeddings = np.append(rel_embeddings[1:3], rel_embeddings[4:], axis=0)
            ent_embeddings = []
            entity = []

            for rel in val_embeddings:
                val_count = len(graph.relation_value_map[rel])
                entity.extend(graph.relation_value_map[rel])
                ent_embeddings.extend(val_embeddings[rel][:val_count])

            assert len(ent_embeddings) == len(entity)

            rltranse.close_tf_session()
        elif params['ea_method'] in [VEER]:
            veer = VEER(model, columns, dimension=ea_params['dimension'],
                        learning_rate=ea_params['learning_rate'],
                        margin=ea_params['margin'],
                        regularizer_scale=ea_params['regularizer_scale'],
                        batchSize=ea_params['batchSize'])
            try:
                veer.restore_model(self._get_tf_model_filename(dataset, veer))
            except Exception as e:
                logger.error(e)
                #Train Model
                loss, val_loss = veer.train(max_epochs=ea_params['epochs'])
                logger.info("Training Complete with loss: %f, val_loss:%f", loss, val_loss)
                veer.save_model(self._get_tf_model_filename(dataset, veer))

            ent_embeddings = veer.get_val_embeddings()
            rel_embeddings = None
            entity = veer.get_values()
            veer.close_tf_session()
        else:
            raise Exception("Unknown Entity Alignment method")

        #Train WERL weights
        werl = WERL(model, columns, entity, ent_embeddings, rel_embeddings,
                        learning_rate=params['learning_rate'],
                        margin=params['margin'],
                        regularizer_scale=params['regularizer_scale'],
                        batchSize=params['batchSize'])
        loss, val_loss = werl.train(max_epochs=params['epochs'])
        logger.info("Training Complete with loss: %f, val_loss:%f", loss, val_loss)

        #Test Model
        result_prob, accuracy = werl.test()
        logger.info("Predict count: %d", len(result_prob))
        logger.info("Sample Prob: %s", str([ (c, (a, b) in dataset.true_test_links)
                                        for (a,b,c) in result_prob[:20]]))
        logger.info("Column Weights: %s", str(werl.get_col_weights()))
        logger.info("Accuracy: %s", str(accuracy))

        #Compute Performance measures
        optimal_threshold, max_fscore = get_optimal_threshold(result_prob, dataset.true_test_links, max_threshold=2.0)

        try:
            params['threshold'] = optimal_threshold
            result = pd.MultiIndex.from_tuples([(e1, e2) for (e1, e2, d) in result_prob if d <= optimal_threshold])
            log_quality_results(logger, result, dataset.true_test_links, len(dataset.test_links), params)
        except Exception as e:
            logger.info("Zero Reults")
            logger.error(e)

        #Test Model
        logger = get_logger('RL.Test.MERL.' + str(dataset))
        result_prob, accuracy = werl.test_merl()
        logger.info("Predict count: %d", len(result_prob))
        logger.info("Sample Prob: %s", str([ (c, (a, b) in dataset.true_test_links)
                                        for (a,b,c) in result_prob[:20]]))
        logger.info("Column Weights: %s", str(werl.get_col_weights()))
        logger.info("Accuracy: %s", str(accuracy))

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
        #ir_metrics = InformationRetrievalMetrics(result_prob, dataset.true_test_links)
        precison_at_1 = None#ir_metrics.log_metrics(logger, params)

        #Test Without Weights = Mean Emebedding for Record Linkage
        logger = get_logger('RL.Test.NoWT.' + str(dataset))

        result_prob, accuracy = werl.test_without_weight()
        logger.info("Predict count: %d", len(result_prob))
        logger.info("Sample Prob: %s", str([ (c, (a, b) in dataset.true_test_links)
                                        for (a,b,c) in result_prob[:20]]))
        logger.info("Column Weights: %s", str(werl.get_col_weights()))
        logger.info("Accuracy: %s", str(accuracy))

        #Compute Performance measures
        optimal_threshold, nowt_max_fscore = get_optimal_threshold(result_prob, dataset.true_test_links, max_threshold=2.0)

        try:
            params['threshold'] = optimal_threshold
            result = pd.MultiIndex.from_tuples([(e1, e2) for (e1, e2, d) in result_prob if d <= optimal_threshold])
            log_quality_results(logger, result, dataset.true_test_links, len(dataset.test_links), params)
        except Exception as e:
            logger.info("Zero Reults")
            logger.error(e)

        #Log MAP, MRR and Hits@K
        #ir_metrics = InformationRetrievalMetrics(result_prob, dataset.true_test_links)
        #nowt_precison_at_1 = ir_metrics.log_metrics(logger, params)
        werl.close_tf_session()

        return (max_fscore, precison_at_1)

    def get_default_params(self):
        #Params for WERL method only. For ea_params, check get_optimal_ea_params
        return {'learning_rate': 0.1, 'margin': 1, 'epochs': 50, 'regularizer_scale' : 0.1,
                'batchSize' : 64,  'ea_method' : RLTransE}

    def test_cora(self):
        return self._test_werl(Cora, ['title', 'author', 'publisher', 'date',
                        'pages', 'volume', 'journal', 'address'], self.get_default_params())

    def test_febrl(self):
        self._test_werl(FEBRL, ['given_name', 'surname', 'state', 'date_of_birth', 'postcode'],
                self.get_default_params())

    def test_census(self):
        self._test_werl(Census, ['Noms_harmo', 'cognom_1', 'cohort', 'estat_civil',
                    'parentesc_har', 'ocupacio_hisco'], self.get_default_params())

    def _test_grid_search(self, dataset, columns):
        dimension= [16, 32, 64]
        batchSize= [32, 64]
        learning_rate= [0.1]
        margin= [1, 0.1]
        regularizer_scale = [0.1]
        epochs = [50, 100, 500]
        neg_rate = [7]
        neg_rel_rate = [1]
        ea_method = [TransE, TransH]
        count = 0
        max_fscore = 0
        max_prec_at_1 = 0

        model = dataset()
        logger = get_logger('RL.Test.GridSearch.WERL.' + str(model))

        for d, bs, lr, m, reg, e, nr, nrr, ea in itertools.product(dimension, batchSize,
                learning_rate, margin, regularizer_scale, epochs, neg_rate, neg_rel_rate, ea_method):
            params = {'learning_rate': lr, 'margin': m, 'dimension': d, 'epochs': e, 'batchSize' : bs,
                        'regularizer_scale' : reg, 'neg_rate' : nr, 'neg_rel_rate': nrr, 'ea_method' : ea}
            logger.info("\nTest:%d, PARAMS: %s", count, str(params))
            count = count + 1
            cur_fscore, cur_prec_at_1 = self._test_werl(dataset, columns, params)
            if max_fscore <= cur_fscore:
                max_fscore = cur_fscore
            if max_prec_at_1 <= cur_prec_at_1:
                max_prec_at_1 = cur_prec_at_1
            logger.info("Ran total %d Tests.", count)
            logger.info("Max Fscore: %f", max_fscore)
            logger.info("Max Mean Precision@1: %f", max_prec_at_1)

    def test_grid_search_cora(self):
        self._test_grid_search(Cora, ['title', 'author', 'publisher', 'date',
                        'pages', 'volume', 'journal', 'address'])

    def test_grid_search_febrl(self):
        self._test_grid_search(FEBRL, ['given_name', 'surname', 'state', 'date_of_birth', 'postcode'])

    def test_grid_search_census(self):
        self._test_grid_search(Census, ['Noms_harmo', 'cognom_1', 'cohort', 'estat_civil',
                    'parentesc_har', 'ocupacio_hisco'])

    def get_optimal_ea_params(self, model, ea_method):
        params = {'learning_rate': 0.1, 'margin': 1, 'dimension': 128 , 'epochs': 1000,
            'regularizer_scale' : 0.1, 'batchSize' : 32, 'neg_rate' : 7, 'neg_rel_rate': 1}
        if model == Census:
            if ea_method in [TransE, TransH]:
                params['dimension'] = 80
                params['batchSize'] = 100
                params['epochs'] = 50
                if ea_method == TransH:
                    params['epochs'] = 500
            elif ea_method == RLTransE:
                params['dimension'] = 64
                params['batchSize'] = 64
                params['epochs'] = 5000
            elif ea_method == VEER:
                params['dimension'] = 16
                params['batchSize'] = 32
                params['epochs'] = 100
                params['margin'] = 0.1
        elif model == FEBRL:
            if ea_method in [TransE]:
                params['dimension'] = 80
                params['batchSize'] = 100
                params['epochs'] = 500
            elif ea_method == TransH:
                params['dimension'] = 64
                params['batchSize'] = 128
                params['epochs'] = 1000
            elif ea_method == RLTransE:
                params['dimension'] = 128
                params['batchSize'] = 32
                params['epochs'] = 5000
                params['margin'] = 0.1
            elif ea_method == VEER:
                params['dimension'] = 16
                params['batchSize'] = 32
                params['epochs'] = 50
        elif model == Cora:
            if ea_method == TransE:
                params['dimension'] = 80
                params['batchSize'] = 100
                params['epochs'] = 50
            elif ea_method == TransH:
                params['dimension'] = 64
                params['batchSize'] = 128
                params['epochs'] = 1000
            elif ea_method == RLTransE:
                params['dimension'] = 32
                params['batchSize'] = 32
                params['epochs'] = 1000
            elif ea_method == VEER:
                params['dimension'] = 16
                params['batchSize'] = 64
                params['epochs'] = 500
                params['margin'] = 0.1
        return params