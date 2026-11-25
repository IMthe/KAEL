"""Feature-engineering agent: random-forest ranking of descriptors."""
import numpy as np
from sklearn.ensemble import RandomForestRegressor

LABELS = {"stacking": "Stacking type", "phi": "Cationic potential", "chi": "Electronegativity",
          "nv": "Li-vacancy content"}


def rank(rows, seed=0, trees=500):
    """Impurity importance of stacking (one-hot, summed), phi, chi and nv for conductivity."""
    stack = sorted({r["stacking"] for r in rows})
    x = np.column_stack([[r[k] for r in rows] for k in ("phi", "chi", "nv")] +
                        [[float(r["stacking"] == s) for r in rows] for s in stack])
    y = np.array([r["sigma"] for r in rows])
    rf = RandomForestRegressor(n_estimators=trees, max_features="sqrt", min_samples_leaf=2,
                               random_state=seed, oob_score=True).fit(x, y)
    imp = rf.feature_importances_
    scores = {"phi": imp[0], "chi": imp[1], "nv": imp[2], "stacking": imp[3:].sum()}
    return dict(n=len(rows), oob_r2=float(rf.oob_score_),
                importance={k: float(v) for k, v in sorted(scores.items(), key=lambda kv: -kv[1])})


def plot(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    items = list(result["importance"].items())[::-1]
    fig, ax = plt.subplots(figsize=(3.6, 2.2))
    ax.barh([LABELS[k] for k, _ in items], [v for _, v in items], color="#3d85c6")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
