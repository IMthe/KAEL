"""Model fitting across the workflow, figures and a plain-text summary."""
import numpy as np

from . import action, fitting, forms


def panel_rows(source, subset, literature, measured):
    rows = literature + (measured if source == "combined" else [])
    return [r for r in rows if r["delta_li"] == 0] if subset == "vacancy_free" else rows


def peak(model):
    """Maximum of the ccp expression, offset + amplitude, at its fitted peak coordinates."""
    o, a, _, phi0, _, nv0 = model["ccp"]["params"]
    return dict(sigma=o + a, phi=phi0, nv=nv0)


def fit_all(literature, measured, starts=64):
    out = dict(cohort=dict(literature=len(literature), experiments=len(measured),
                           hcp=sum(r["stacking"] == "hcp" for r in literature),
                           ccp=sum(r["stacking"] == "ccp" for r in literature)))
    lit = fitting.fit_phases(forms.LITERATURE, literature, starts, loo=True)
    out["literature_model"] = lit
    for key, (source, subset, fm) in forms.PANELS.items():
        rows = panel_rows(source, subset, literature, measured)
        out[f"panel_{key}"] = fitting.fit_phases(fm, rows, starts, loo=key == "e")
    final = out["panel_e"]
    final["ceiling"] = peak(final["model"])
    best = max(measured, key=lambda r: r["sigma"])
    final["best_measured"] = dict(id=best["id"], formula=best["formula"], sigma=best["sigma"],
                                  fraction_of_ceiling=best["sigma"] / final["ceiling"]["sigma"])
    return out


def figures(result, literature, measured, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.4))
    for ax, key in zip(axes.flat, "abcdef"):
        source, subset, _ = forms.PANELS[key]
        rows = panel_rows(source, subset, literature, measured)
        pred = result[f"panel_{key}"]["predictions"]
        for stacking, colour in (("hcp", "#1f3c88"), ("ccp", "#6fa8dc")):
            g = [r for r in rows if r["stacking"] == stacking and not r["id"].startswith("E")]
            ax.scatter([r["sigma"] for r in g], [pred[r["id"]] for r in g], s=14, c=colour, label=stacking)
        e = [r for r in rows if r["id"].startswith("E")]
        if e:
            ax.scatter([r["sigma"] for r in e], [pred[r["id"]] for r in e], marker="*", s=60, c="#cc0000", label="this work")
        ax.plot([-0.5, 4.5], [-0.5, 4.5], "k--", lw=0.8)
        m = result[f"panel_{key}"]["metrics"]
        ax.set_title(f"({key})  R² within phase = {m['within_phase_r2']:.4f}", fontsize=9)
        ax.set_xlabel("Measured σ (mS cm⁻¹)", fontsize=8)
        ax.set_ylabel("Fitted σ (mS cm⁻¹)", fontsize=8)
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(folder / "fig3_parity.png", dpi=300)
    plt.close(fig)

    model = result["panel_e"]["model"]
    phi, nv = np.meshgrid(np.linspace(1.70, 1.84, 281), np.linspace(-0.05, 0.95, 201))
    grid = {"phi": phi.ravel(), "nv": nv.ravel(), "chi": np.zeros(phi.size)}
    from .expressions import Expression
    ccp = Expression(model["ccp"]["expression"])(model["ccp"]["params"], grid).reshape(phi.shape)
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    cs = ax.contourf(phi, nv, ccp, 30, cmap="viridis")
    fig.colorbar(cs, label="ccp σ (mS cm⁻¹)")
    ax.scatter([r["phi"] for r in measured], [r["nv"] for r in measured], c="w", s=12, edgecolors="k")
    c = result["panel_e"]["ceiling"]
    ax.scatter([c["phi"]], [c["nv"]], marker="*", s=120, c="#cc0000", clip_on=False)
    ax.set_xlabel("Cationic potential φ")
    ax.set_ylabel("Li-vacancy content nᵥ")
    fig.tight_layout()
    fig.savefig(folder / "fig3_surface.png", dpi=300)
    plt.close(fig)


def summary(done):
    lines = []
    if "rank" in done:
        r = done["rank"]
        lines.append(f"Descriptor ranking ({r['n']} records, OOB R² {r['oob_r2']:.3f}): " +
                     ", ".join(f"{k} {v:.3f}" for k, v in r["importance"].items()))
    if "fit" in done:
        f = done["fit"]
        lit, fin = f["literature_model"], f["panel_e"]
        lines.append(f"Cohort: {f['cohort']['literature']} literature records "
                     f"({f['cohort']['hcp']} hcp, {f['cohort']['ccp']} ccp) + {f['cohort']['experiments']} experiments")
        lines.append(f"Literature model: within-phase R² {lit['metrics']['within_phase_r2']:.4f}, "
                     f"LOO {lit['loo_metrics']['within_phase_r2']:.4f}; ccp optimum at phi {lit['model']['ccp']['params'][3]:.4f}")
        for k in "abcdef":
            m = f[f"panel_{k}"]["metrics"]
            lines.append(f"Panel {k}: n={m['n']}, within-phase R² {m['within_phase_r2']:.4f}, MAE {m['mae']:.4f}, BIC {m['bic']:.1f}")
        lines.append(f"Final model: LOO within-phase R² {fin['loo_metrics']['within_phase_r2']:.4f}; ccp maximum "
                     f"{fin['ceiling']['sigma']:.4f} mS/cm at phi {fin['ceiling']['phi']:.4f}, nv {fin['ceiling']['nv']:.3f}; "
                     f"best measured {fin['best_measured']['sigma']} mS/cm = "
                     f"{100 * fin['best_measured']['fraction_of_ceiling']:.1f}% of the maximum")
    if "recommend" in done:
        lines.append("Next batch: " + "; ".join(f"{r['formula']} ({r['stacking']}, surface {r['surface']:.2f})"
                                                for r in done["recommend"]))
    if "discover" in done:
        for s, d in done["discover"].items():
            if d["selected"]:
                lines.append(f"{s}: {d['selected']['expression']}  MSE {d['selected']['mse']:.4g}")
    return "\n".join(lines)
