"""Synthetic demonstration data in the input format (not experimental data)."""
import csv

import numpy as np

from .data import descriptors

LIT_FIELDS = ["id", "formula", "stacking", "c", "nv", "delta_li", "phi", "chi", "cation_radius", "anion_radius", "sigma"]
EXP_FIELDS = ["id", "iteration", "formula", "stacking", "c", "nv", "phi", "chi", "sigma"]


def surface(phi, nv, stacking):
    if stacking == "ccp":
        return 1.1 + 2.4 * np.exp(-45 * abs(phi - 1.765) - 2.5 * abs(nv - 0.2))
    return 0.5 * np.exp(-20 * abs(phi - 1.76) - 3 * abs(nv - 0.25))


def record(ident, c, er, rng, noise=0.05):
    d = descriptors(c, er)
    stacking = "ccp" if d["phi"] > 1.755 else "hcp"
    sigma = max(0.01, surface(d["phi"], d["nv"], stacking) * (1 + rng.normal(0, noise)))
    return dict(id=ident, formula=d["formula"], stacking=stacking, c=c, nv=round(3 - c, 4), delta_li=round(3 - c, 4),
                phi=round(d["phi"], 5), chi=round(d["chi"], 5), cation_radius=round(0.94 + 0.09 * er, 4),
                anion_radius=1.67, sigma=round(sigma, 4))


def write(folder, seed=0):
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    lit = [record(f"S{i:03d}", float(rng.choice([2.7, 2.8, 2.9, 3.0])), float(rng.uniform(0, 1)), rng)
           for i in range(40)]
    plan = [(1, 3.0, 1.0), (1, 3.0, 0.0), (2, 3.0, 0.5), (3, 2.8, 0.5)]
    exp = []
    for k, (it, c, er) in enumerate(plan, 1):
        r = record(f"X{k:02d}", c, er, rng)
        exp.append({f: (it if f == "iteration" else r[f]) for f in EXP_FIELDS})
    for name, rows, fields in (("literature.csv", lit, LIT_FIELDS), ("experiments.csv", exp, EXP_FIELDS)):
        with open(folder / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fields)
            w.writeheader()
            w.writerows(rows)
    return folder / "literature.csv", folder / "experiments.csv"
