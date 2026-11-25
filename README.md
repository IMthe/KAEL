# KAEL — Knowledge-Action Exploratory Learning

KAEL couples a knowledge-discovery agent, which turns literature data into an interpretable conductivity expression, with an action agent, which selects experiments in the candidate composition space and feeds the measurements back into the expression.

## Workflow

| Step | Module |
|---|---|
| Random-forest ranking of stacking type, cationic potential, electronegativity and Li-vacancy content | `kael/ranking.py` |
| LLM-guided symbolic regression: the LLM proposes phase-specific expressions, multistart least squares fits every coefficient, and the Pareto front of error and complexity is resolved by BIC | `kael/knowledge.py` |
| Fitting of the selected expressions to literature and experimental data, leave-one-out validation, BIC and the conductivity maximum | `kael/fitting.py`, `kael/report.py` |
| Gaussian-process Bayesian optimisation with the symbolic model as prior mean and expected improvement as acquisition | `kael/action.py` |

All coefficients of the conductivity expressions, including the peak position in cationic potential and Li-vacancy content and the peak height, are free parameters of the least-squares fit; only the decay rates of the exponential terms are non-negative. The conductivity maximum is the peak of the fitted ccp expression.

## Installation

Python 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Demo

```bash
python -m kael demo --out demo_results
```

The demo writes a synthetic literature table (40 records) and a synthetic experiment table (4 records) generated from a known conductivity surface, then runs the full workflow. It finishes in under a minute on a laptop and recovers the peak of the generating surface. The summary is printed to the console and saved in `demo_results/demo.json`, together with descriptor-ranking, parity and conductivity-surface plots.

## Usage

```bash
python -m kael run --literature literature.csv --experiments experiments.csv --out results
python -m kael fit --literature literature.csv --experiments experiments.csv --out results
python -m kael recommend --literature literature.csv --experiments experiments.csv --batch 2
python -m unittest discover tests
```

`run` performs the descriptor ranking, all model fits and the next-batch recommendation (about 10 min with the default 128 optimiser starts). The knowledge-discovery agent calls an LLM through the OpenAI Responses API:

```bash
export OPENAI_API_KEY=...                   # Windows PowerShell: $env:OPENAI_API_KEY="..."
export OPENAI_MODEL=gpt-5
python -m kael discover --literature literature.csv --experiments experiments.csv --rounds 5
```

New measurements are added as rows of the experiment table; the next `recommend` or `fit` uses them.

## Input format

| Table | Columns |
|---|---|
| Literature | `id, formula, stacking, c, nv, delta_li, phi, chi, cation_radius, anion_radius, sigma` |
| Experiments | `id, iteration, formula, stacking, c, nv, phi, chi, sigma` |

`stacking` is `hcp`, `ccp`, `mixed` or `other`; `c` is the Li content per six halides, `nv = 3 − c`, `delta_li` the Li deficit relative to the parent compound, `phi` the cationic potential, `chi` the stoichiometry-weighted mean Pauling electronegativity, and `sigma` the room-temperature ionic conductivity in mS cm⁻¹.

The modelling cohort is selected from the literature table as Li₃MX₆-type records: a single halide X₆ (Cl, Br or I), trivalent M, a single hcp or ccp stacking and a Li content of 2.7–3.3 per formula unit (`kael/data.py`). The candidate space is the Er–In–Cl feed grid of ten Li contents (2.1–3.0) and ten In and ten Er feed levels, with cationic potential and electronegativity computed from Shannon radii and Pauling electronegativities.

## Outputs

`<out>/<command>.json` holds all parameters, predictions, metrics and recommendations. Random seeds are fixed, so repeated runs give identical results.

## License

MIT (see `LICENSE`).
