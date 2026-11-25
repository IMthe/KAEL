"""Input tables, the Li3MX6 modelling cohort and the Er-In-Cl candidate space.

Literature table columns: id, formula, stacking (hcp, ccp, mixed or other), c (Li per six halides),
nv (3 - c), delta_li (Li deficit relative to the parent compound), phi, chi, cation_radius,
anion_radius, sigma (room-temperature ionic conductivity, mS/cm).
Experiment table columns: id, iteration, formula, stacking, c, nv, phi, chi, sigma.
"""
import csv
import re

TRIVALENT = {"Sc", "Y", "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm",
             "Yb", "Lu", "In", "Ga", "Al", "Fe", "Cr", "Ti", "Bi"}
HALOGENS = {"Cl", "Br", "I"}
ANIONS = HALOGENS | {"F", "O", "S"}
# Shannon radii (Å, six-fold) and Pauling electronegativities used for the candidate space.
RADIUS = {"Li": 0.90, "Er": 1.03, "In": 0.94, "Cl": 1.67}
PAULING = {"Li": 0.98, "Er": 1.24, "In": 1.78, "Cl": 3.16}
NUMERIC = ("c", "nv", "delta_li", "phi", "chi", "cation_radius", "anion_radius", "cation_charge", "sigma")


def read(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in NUMERIC:
            if r.get(k, "") != "":
                r[k] = float(r[k])
    return rows


def literature(path):
    return read(path)


def experiments(path):
    rows = read(path)
    for r in rows:
        r["iteration"] = int(r["iteration"])
        r["delta_li"] = r["nv"]
    return rows


def composition(formula):
    """Element amounts of a formula such as Li2.8(Er0.6In0.4)1.067Cl6."""
    def expand(m):
        factor = float(m.group(2) or 1)
        return "".join(f"{e}{float(n or 1) * factor:g}" for e, n in re.findall(r"([A-Z][a-z]?)([0-9.]*)", m.group(1)))
    while "(" in formula:
        formula = re.sub(r"\(([^()]*)\)([0-9.]*)", expand, formula)
    amounts = {}
    for e, n in re.findall(r"([A-Z][a-z]?)([0-9.]*)", formula):
        amounts[e] = amounts.get(e, 0.0) + float(n or 1)
    return amounts


def li3mx6(record, li_window=(2.7, 3.3)):
    """Li3MX6-type record: one halide X6, trivalent M, single hcp or ccp stacking, near-Li3 content."""
    amounts = composition(record["formula"])
    anions = [e for e in amounts if e in ANIONS]
    metals = [e for e in amounts if e not in ANIONS and e != "Li"]
    return (len(anions) == 1 and anions[0] in HALOGENS and abs(amounts[anions[0]] - 6) < 1e-6
            and metals and all(m in TRIVALENT for m in metals)
            and record["stacking"] in ("hcp", "ccp")
            and li_window[0] <= record["c"] <= li_window[1])


def cohort(rows):
    return [r for r in rows if li3mx6(r)]


def descriptors(c, er):
    """Cationic potential, Li-vacancy content and mean electronegativity of Li_c(Er_x In_1-x)_{(6-c)/3}Cl6."""
    m = (6 - c) / 3
    phi = (c / RADIUS["Li"] + m * (er * 3 / RADIUS["Er"] + (1 - er) * 3 / RADIUS["In"])) / (6 / RADIUS["Cl"])
    atoms = {"Li": c, "Er": m * er, "In": m * (1 - er), "Cl": 6.0}
    chi = sum(PAULING[e] * n for e, n in atoms.items()) / sum(atoms.values())
    formula = f"Li{c:g}(Er{er:.4f}In{1 - er:.4f}){m:.4f}Cl6"
    return dict(c=c, nv=3 - c, phi=phi, chi=chi, er_fraction=er, formula=formula)


def candidate_space(li_levels=None, feed_levels=None):
    """Feed grid: ten Li levels x ten In feed levels x ten Er feed levels (1,000 settings)."""
    from fractions import Fraction
    li_levels = li_levels or [round(2.1 + 0.1 * i, 1) for i in range(10)]
    feed_levels = feed_levels or [Fraction(i, 10) for i in range(10)]
    pool = {}
    for c in li_levels:
        for f_in in feed_levels:
            for f_er in feed_levels:
                if f_in + f_er == 0:
                    continue
                er = Fraction(f_er) / (Fraction(f_in) + Fraction(f_er))
                key = f"Li{c:g}-Er{er.numerator}/{er.denominator}"
                pool.setdefault(key, dict(id=key, **descriptors(c, float(er))))
    return list(pool.values())
