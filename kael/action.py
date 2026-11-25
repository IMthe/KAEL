"""Action agent: knowledge-informed Gaussian-process Bayesian optimisation over the candidate space."""
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import LogisticRegression

from . import data, fitting, forms


def features(rows):
    return np.column_stack([[r["phi"] for r in rows], [r["nv"] for r in rows]])


def stacking_classifier(labelled):
    """Logistic classifier for hcp/ccp stacking in (phi, nv), trained on labelled records."""
    x = features(labelled)
    mu, sd = x.mean(0), x.std(0)
    clf = LogisticRegression().fit((x - mu) / sd, [r["stacking"] == "ccp" for r in labelled])
    return lambda rows: clf.predict_proba((features(rows) - mu) / sd)[:, 1]


def assign_stacking(pool, labelled):
    p = stacking_classifier(labelled)(pool)
    return [dict(r, stacking="ccp" if q >= 0.5 else "hcp", p_ccp=float(q)) for r, q in zip(pool, p)]


def knowledge_model(literature, measured, starts=32):
    """Final-form phase expressions refitted to literature plus all measurements so far."""
    rows = literature + measured
    return fitting.fit_phases(forms.PANELS["e"][2], rows, starts=starts)["model"]


def measured_key(r):
    amounts = data.composition(r["formula"])
    er, ind = amounts.get("Er", 0.0), amounts.get("In", 0.0)
    return round(r["c"], 2), round(er / (er + ind), 3) if er + ind else None


def recommend(literature, measured, batch=2, starts=32, seed=0):
    """Rank unmeasured candidates by expected improvement over the best measured conductivity."""
    labelled = literature + measured
    pool = assign_stacking(data.candidate_space(), labelled)
    done = {measured_key(r) for r in measured}
    pool = [r for r in pool if (round(r["c"], 2), round(r["er_fraction"], 3)) not in done]
    model = knowledge_model(literature, measured, starts)
    prior = fitting.predict(model, pool)
    x = features(pool)
    mu, sd = x.mean(0), x.std(0)
    z = (x - mu) / sd
    if not measured:
        order = np.argsort(-prior)[:batch]
        return [dict(pool[i], surface=float(prior[i])) for i in order], model
    zm = (features(measured) - mu) / sd
    resid = np.array([r["sigma"] for r in measured]) - fitting.predict(model, measured)
    best = max(r["sigma"] for r in measured)
    kernel = ConstantKernel(0.25) * Matern(length_scale=[1.0, 1.0], nu=2.5) + WhiteKernel(0.01)
    gp = GaussianProcessRegressor(kernel, n_restarts_optimizer=5, random_state=seed).fit(zm, resid)
    chosen, zt, rt = [], zm.copy(), resid.copy()
    for _ in range(batch):
        m, s = gp.predict(z, return_std=True)
        mean = prior + m
        gain = mean - best - 0.01
        ei = gain * norm.cdf(gain / np.maximum(s, 1e-9)) + s * norm.pdf(gain / np.maximum(s, 1e-9))
        ei[chosen] = -np.inf
        i = int(np.argmax(ei))
        chosen.append(i)
        # Kriging believer: condition on the predicted value before choosing the next batch member.
        zt, rt = np.vstack([zt, z[i]]), np.append(rt, m[i])
        gp = GaussianProcessRegressor(gp.kernel_, optimizer=None).fit(zt, rt)
    out = []
    for i in chosen:
        out.append(dict(pool[i], surface=float(prior[i])))
    return out, model

