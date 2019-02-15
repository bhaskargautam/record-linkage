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

        #Begin SEEA iterations, passing true pairs only to debug the alignments.
        results = seea.seea_iterate(entity_pairs, true_pairs, params['beta'],
                                    params['max_iter'], params['max_epochs'])
        try:
            result_pairs = pd.MultiIndex.from_tuples(results)
            log_quality_results(logger, result_pairs, true_pairs, len(entity_pairs), params)
        except Exception as e:
            logger.error(e)
            logger.info("No Aligned pairs found.")

        ent_embeddings = seea.get_ent_embeddings()
        export_embeddings(file_prefix, 'SEEA', entity, ent_embeddings)

        seea.close_tf_session()

    def get_default_params(self):
        return {'beta': 10, 'max_iter' : 10, 'dimension': 80,
                'learning_rate' : 0.1, 'batchSize' : 10, 'margin' : 1,
                'regularizer_scale' : 0.1, 'max_epochs' : 100}

    def test_seea_cora(self):
        self._test_seea(Cora, config.CORA_FILE_PREFIX, self.get_default_params())

    def test_seea_febrl(self):
        self._test_seea(FEBRL, config.FEBRL_FILE_PREFIX, self.get_default_params())

    def test_seea_census(self):
        self._test_seea(Census, config.CENSUS_FILE_PREFIX, self.get_default_params())