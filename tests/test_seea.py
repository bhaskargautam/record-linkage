import config
import pandas as pd
import unittest

from common import export_embeddings, get_logger, log_quality_results, sigmoid
from data.cora import Cora
from data.febrl import FEBRL
from data.census import Census
from EAR.seea import SEEA
from EAR.model import Graph_EAR
from scipy import spatial


logger = get_logger('TestSEEA')

class Test_SEEA(unittest.TestCase):

    def _test_seea(self, dataset, file_prefix, params):
        logger = get_logger('Test_SEEA.' + str(file_prefix))
        try:
            graph = Graph_EAR(file_prefix)
            entity, attribute, relation, value, atriples, rtriples, \
                entity_pairs, true_pairs = graph.load_kg_ear_model()
        except IOError:
            model = dataset()
            entity, attribute, relation, value, atriples, rtriples, \
                entity_pairs, true_pairs = model.get_ear_model()

        seea = SEEA(entity, attribute, relation, value, atriples, rtriples,
                        dimension = params['dimension'],
                        learning_rate = params['learning_rate'],
                        batchSize = params['batchSize'],
                        margin = params['margin'],
                        regularizer_scale = params['regularizer_scale'])
        all_results = []
        total_candidate_pairs = len(entity_pairs)
        for j in range(0, params['max_iter']):
            loss = seea.train(params['max_epochs'])
            logger.info("Training Complete with loss: %f for iteration: %d", loss, j)

            ent_embeddings = seea.get_ent_embeddings()
            results = []
            for i in range(0, len(entity_pairs)):
                distance = abs(spatial.distance.cosine(
                                    ent_embeddings[entity_pairs[i][0]],
                                    ent_embeddings[entity_pairs[i][1]]))
                results.append((entity_pairs[i][0], entity_pairs[i][1], distance))

            results = sorted(results, key=(lambda x: x[2]))
            alligned_pairs = [(e1, e2) for (e1, e2, d) in results[:params['beta']]]
            all_results.extend(alligned_pairs)

            if len(alligned_pairs):
                result_pairs = pd.MultiIndex.from_tuples(all_results)
                log_quality_results(logger, result_pairs, true_pairs, total_candidate_pairs, params)
            else:
                logger.info("No new results")

            for (e1, e2, d) in results[:params['beta']]:
                logger.info("iteration: %d, distance: %f true_pairs: %s", j, d, (e1, e2) in true_pairs)
                entity_pairs.remove((e1, e2))

            new_triples = [(e1, e2, len(relation) - 1) for (e1, e2) in alligned_pairs]
            seea.add_rel_triples(new_triples)

        ent_embeddings = seea.get_ent_embeddings()
        export_embeddings(file_prefix, 'TransE', entity, ent_embeddings)

        seea.close_tf_session()

    def get_default_params(self):
        return {'beta': config.SEEA_BETA, 'max_iter' : 100, 'dimension': 80,
                'learning_rate' : 0.1, 'batchSize' : 100, 'margin' : 1,
                'regularizer_scale' : 0.1, 'max_epochs' : 100}

    def test_seea_cora(self):
        self._test_seea(Cora, config.CORA_FILE_PREFIX, self.get_default_params())

    def test_seea_febrl(self):
        self._test_seea(FEBRL, config.FEBRL_FILE_PREFIX, self.get_default_params())

    def test_seea_census(self):
        self._test_seea(Census, config.CENSUS_FILE_PREFIX, self.get_default_params())