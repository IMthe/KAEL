"""Multistart least-squares fitting of phase-specific expressions and model statistics."""
import numpy as np
from scipy.optimize import least_squares

from .expressions import Expression


def table(rows, keys=("phi", "nv", "chi")):
    return {k: np.array([r[k] for r in rows], float) for k in keys}


def random_start(roles, rows, rng):
    """Draw one starting point from broad, data-scaled distributions (no bounds are imposed in the fit)."""
    y = np.array([r["sigma"] for r in rows])
    x = []
    for role in roles:
        kind, _, var = role.partition(":")
        if kind == "center":
            v = np.array([r[var] for r in rows])
            x.append(rng.choice(v) + rng.normal(0, 0.1 * max(np.ptp(v), 1e-3)))
        elif kind == "rate":
            x.append(np.exp(rng.uniform(np.log(1), np.log(200))) if var == "phi" else np.exp(rng.uniform(np.log(0.05), np.log(30))))
        elif kind == "amplitude":
            x.append(rng.uniform(0.05, 2) * y.max())
        elif kind == "offset":
            x.append(rng.uniform(0, 1) * y.mean())
        else:
            x.append(rng.normal(0, 3))
    return np.array(x)


def fit(source, rows, roles=None, initial=None, starts=64, seed=0):
    """Minimise the sum of squared conductivity residuals from many starting points."""
    expr = Expression(source)
    data, y = table(rows), np.array([r["sigma"] for r in rows], float)
    rng = np.random.default_rng(seed)

    def residual(p):
        r = expr(p, data) - y
        return np.where(np.isfinite(r), r, 1e6)

    guesses = []
    if initial is not None:
        initial = np.asarray(initial, float)
        guesses.append(initial)
        guesses += [initial * np.exp(rng.normal(0, 0.5, initial.size)) for _ in range(starts // 2)]
    while len(guesses) < starts:
        guesses.append(random_start(roles, rows, rng) if roles else rng.normal(0, 3, expr.n_params))
    # Decay rates of exp(-k|x - x0|) terms are non-negative; all other coefficients are unbounded.
    lower = np.array([0.0 if (roles and roles[i].startswith("rate")) else -np.inf for i in range(expr.n_params)])
    best = None
    for g in guesses:
        g = np.maximum(g, np.where(np.isfinite(lower), lower + 1e-6, -np.inf))
        try:
            sol = least_squares(residual, g, method="trf", bounds=(lower, np.inf), max_nfev=4000)
        except (ValueError, FloatingPointError):
            continue
        if np.all(np.isfinite(sol.x)) and (best is None or sol.cost < best.cost):
            best = sol
    if best is None:
        raise ValueError("No converged fit")
    pred = expr(best.x, data)
    return dict(expression=source, params=best.x.tolist(), n=len(rows), nodes=expr.nodes,
                mse=float(np.mean((pred - y) ** 2)), predictions=pred.tolist())


def metrics(y, pred, stacking, n_params=None):
    y, pred, stacking = np.asarray(y, float), np.asarray(pred, float), np.asarray(stacking)
    sse = float(np.sum((pred - y) ** 2))
    within = sum(float(np.sum((y[stacking == s] - y[stacking == s].mean()) ** 2)) for s in set(stacking))
    out = dict(n=int(len(y)), r2=1 - sse / float(np.sum((y - y.mean()) ** 2)),
               within_phase_r2=1 - sse / within, mae=float(np.mean(np.abs(pred - y))))
    if n_params:
        out["bic"] = len(y) * np.log(sse / len(y)) + n_params * np.log(len(y))
    return out


def fit_phases(forms, rows, starts=64, seed=0, loo=False):
    """Fit one expression per stacking type; optionally add leave-one-out predictions."""
    model, pred, loo_pred = {}, {}, {}
    for stacking, (source, roles) in forms.items():
        group = [r for r in rows if r["stacking"] == stacking]
        if not group:
            continue
        result = fit(source, group, roles, starts=starts, seed=seed)
        model[stacking] = dict(expression=source, params=result["params"], n=len(group))
        pred.update(zip([r["id"] for r in group], result["predictions"]))
        if loo:
            expr = Expression(source)
            for i, r in enumerate(group):
                rest = group[:i] + group[i + 1:]
                sub = fit(source, rest, roles, initial=result["params"], starts=max(8, starts // 4), seed=seed + i + 1)
                loo_pred[r["id"]] = float(expr(sub["params"], table([r]))[0])
    y = [r["sigma"] for r in rows]
    s = [r["stacking"] for r in rows]
    k = sum(len(m["params"]) for m in model.values())
    out = dict(model=model, metrics=metrics(y, [pred[r["id"]] for r in rows], s, k),
               predictions={k: float(v) for k, v in pred.items()})
    if loo:
        out["loo_metrics"] = metrics(y, [loo_pred[r["id"]] for r in rows], s)
        out["loo_predictions"] = loo_pred
    return out


def predict(model, rows):
    out = np.full(len(rows), np.nan)
    for stacking, m in model.items():
        idx = [i for i, r in enumerate(rows) if r["stacking"] == stacking]
        if idx:
            out[idx] = Expression(m["expression"])(m["params"], table([rows[i] for i in idx]))
    return out
