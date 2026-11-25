"""Knowledge-discovery agent: LLM-guided symbolic regression with Pareto and BIC selection."""
import math

from . import fitting
from .llm import NUMBER, STRING, ask, obj

PROPOSAL = obj(expression=STRING, initial={"type": "array", "items": NUMBER}, rationale=STRING)

PROMPT = (
    "You are the knowledge-discovery agent of KAEL for halide solid-state electrolytes. Propose ONE compact, "
    "physically interpretable expression for room-temperature ionic conductivity (mS/cm) of a single stacking type. "
    "Descriptors: phi (cationic potential), nv (Li-vacancy content per formula unit, 3 minus Li per six halides). "
    "Syntax: + - * / **, unary minus, exp(), abs(), numeric literals and free parameters p0, p1, ... numbered "
    "consecutively. Use at most six parameters and give one starting value for each. Base the expression on Li+ "
    "transport physics in halides (stacking, cation-anion interaction, carrier and vacancy concentration) and on the "
    "supplied records. Use the feedback on previous proposals to try a structurally different expression each round. "
    "Explain the mechanism in one or two sentences.")


def pareto(models):
    return [m for m in models if not any(o["mse"] <= m["mse"] and o["nodes"] <= m["nodes"]
                                         and (o["mse"] < m["mse"] or o["nodes"] < m["nodes"]) for o in models)]


def bic(m):
    return m["n"] * math.log(max(m["mse"], 1e-300)) + len(m["params"]) * math.log(m["n"])


def discover(rows, rounds=5, starts=32, importance=None):
    """Run the propose-fit-select loop separately for each stacking type."""
    out = {}
    for stacking in ("hcp", "ccp"):
        group = [r for r in rows if r["stacking"] == stacking]
        records = [{k: r[k] for k in ("id", "phi", "nv", "sigma")} for r in group]
        history, fitted = [], []
        for _ in range(rounds):
            proposal = ask(PROMPT, dict(stacking=stacking, records=records, descriptor_importance=importance,
                                        previous=history), PROPOSAL, "kael_expression")
            try:
                m = fitting.fit(proposal["expression"], group, initial=proposal["initial"], starts=starts)
                m.update(rationale=proposal["rationale"], bic=None)
                m["bic"] = bic(m)
                fitted.append(m)
                history.append({k: m[k] for k in ("expression", "mse", "nodes", "bic")})
            except (ValueError, SyntaxError, TypeError, ZeroDivisionError) as exc:
                history.append(dict(expression=proposal["expression"], error=str(exc)))
        front = pareto(fitted)
        out[stacking] = dict(candidates=fitted, pareto=front,
                             selected=min(front, key=lambda m: (m["bic"], m["nodes"])) if front else None)
    return out
