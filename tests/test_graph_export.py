import unittest

from common import get_logger
from data.cora import Cora
from data.febrl import FEBRL
from data.census import Census
from ER.model import Graph_ER
from EAR.model import Graph_EAR
from ERER.model import Graph_ERER

logger = get_logger('TestGraphExport')

class TestGraphExport(unittest.TestCase):

    def _test_er_graph_export(self, dataset):
        model = dataset()
        entity, relation, triples, entity_pairs, true_pairs = model.get_er_model()
        graph = Graph_ER(str(model))
        graph.export_kg_er_model(entity, relation, triples, entity_pairs, true_pairs)
        e,r,t,ep,tp = graph.load_kg_er_model()
        self.assertEqual(len(e), len(entity))
        self.assertEqual(len(r), len(relation))
        self.assertEqual(len(t), len(triples))
        self.assertEqual(len(ep), len(entity_pairs))
        self.assertEqual(len(tp), len(true_pairs))

    def test_er_cora(self):
        self._test_er_graph_export(Cora)

    def test_er_febrl(self):
        self._test_er_graph_export(FEBRL)

    def test_er_census(self):
        self._test_er_graph_export(Census)

    def _test_ear_graph_export(self, dataset):
        model = dataset()
        entity, attribute, relation, value, atriples, rtriples, entity_pairs, true_pairs = model.get_ear_model()
        graph = Graph_EAR(str(model))
        graph.export_kg_ear_model(entity, attribute, relation, value, atriples, rtriples, entity_pairs, true_pairs)
        e,a,r,v,at,rt,ep,tp = graph.load_kg_ear_model()
        self.assertEqual(len(e), len(entity))
        self.assertEqual(len(a), len(attribute))
        self.assertEqual(len(r), len(relation))
        self.assertEqual(len(v), len(value))
        self.assertEqual(len(at), len(atriples))
        self.assertEqual(len(rt), len(rtriples))
        self.assertEqual(len(ep), len(entity_pairs))
        self.assertEqual(len(tp), len(true_pairs))

    def test_ear_cora(self):
        self._test_ear_graph_export(Cora)

    def test_ear_febrl(self):
        self._test_ear_graph_export(FEBRL)

    def test_ear_census(self):
        self._test_ear_graph_export(Census)

    def _test_erer_graph_export(self, dataset):
        model = dataset()
        entA, entB, relA, relB, triA, triB, entity_pairs, prior_pairs, true_pairs = model.get_erer_model()
        graph = Graph_ERER(str(model))
        graph.export_kg_erer_model(entA, entB, relA, relB, triA, triB, entity_pairs, prior_pairs, true_pairs)
        eA, eB, rA, rB, tA, tB, ep, pp, tp = graph.load_kg_erer_model()
        self.assertEqual(len(eA), len(entA))
        self.assertEqual(len(eB), len(entB))
        self.assertEqual(len(rA), len(relA))
        self.assertEqual(len(rB), len(relB))
        self.assertEqual(len(tA), len(triA))
        self.assertEqual(len(tB), len(triB))
        self.assertEqual(len(ep), len(entity_pairs))
        self.assertEqual(len(pp), len(prior_pairs))
        self.assertEqual(len(tp), len(true_pairs))

    def test_erer_cora(self):
        self._test_erer_graph_export(Cora)

    def test_erer_febrl(self):
        self._test_erer_graph_export(FEBRL)

    def test_erer_census(self):
        self._test_erer_graph_export(Census)
