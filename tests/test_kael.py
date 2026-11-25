"""Checks of parsing, fitting and the action agent on synthetic data. Run: python -m unittest discover tests"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from kael import action, data, demo, fitting, forms, ranking
from kael.expressions import Expression


class KAELTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        lit, exp = demo.write(Path(cls.tmp.name))
        cls.lit, cls.exp = data.cohort(data.literature(lit)), data.experiments(exp)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_expression_safety(self):
        with self.assertRaises(ValueError):
            Expression("__import__('os').system('ls')")
        e = Expression("p0*exp(-p1*abs(phi-p2))")
        self.assertEqual(e.n_params, 3)
        self.assertAlmostEqual(float(e([2, 10, 1.76], {"phi": np.array([1.76])})[0]), 2.0)

    def test_cohort_and_candidates(self):
        self.assertTrue(self.lit and all(data.li3mx6(r) for r in self.lit))
        self.assertAlmostEqual(data.descriptors(2.8, 0.6)["phi"], 1.764, places=3)
        self.assertTrue(500 < len(data.candidate_space()) <= 1000)

    def test_free_fit_recovers_peak(self):
        rng = np.random.default_rng(1)
        rows = [dict(id=str(i), stacking="ccp", phi=p, nv=v, chi=2.3,
                     sigma=1 + 2.5 * np.exp(-40 * abs(p - 1.780) - 3 * abs(v - 0.12)) + rng.normal(0, 0.01))
                for i, (p, v) in enumerate(zip(rng.uniform(1.74, 1.80, 40), rng.uniform(0, 0.4, 40)))]
        source, roles = forms.CCP_PHI_NV
        fit = fitting.fit(source, rows, roles, starts=24)
        self.assertAlmostEqual(fit["params"][3], 1.780, places=2)
        self.assertAlmostEqual(fit["params"][5], 0.12, places=1)

    def test_ranking_and_recommendation(self):
        self.assertEqual(set(ranking.rank(self.lit)["importance"]), {"stacking", "phi", "chi", "nv"})
        picks, _ = action.recommend(self.lit, self.exp, batch=2, starts=8)
        self.assertEqual(len(picks), 2)


if __name__ == "__main__":
    unittest.main()
