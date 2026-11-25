"""KAEL command line: python -m kael {demo,rank,discover,fit,recommend,run}."""
import argparse
import json
import warnings
from pathlib import Path

from sklearn.exceptions import ConvergenceWarning

from . import action, data, demo, knowledge, ranking, report


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    p = argparse.ArgumentParser(prog="kael", description=__doc__)
    p.add_argument("command", choices=("demo", "rank", "discover", "fit", "recommend", "run"))
    p.add_argument("--literature", type=Path, help="literature table (CSV)")
    p.add_argument("--experiments", type=Path, help="experiment table (CSV)")
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--starts", type=int, default=128, help="optimiser starting points per fit")
    p.add_argument("--rounds", type=int, default=5, help="LLM proposals per stacking type")
    p.add_argument("--batch", type=int, default=2, help="compositions per active-learning iteration")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    command = args.command
    if command == "demo":
        args.literature, args.experiments = demo.write(args.out / "demo_data")
        args.starts, command = min(args.starts, 32), "run"
    if not (args.literature and args.experiments):
        p.error("--literature and --experiments are required")
    lit = data.cohort(data.literature(args.literature))
    exp = data.experiments(args.experiments)
    done = {}
    if command in ("rank", "run"):
        done["rank"] = ranking.rank(lit)
        ranking.plot(done["rank"], args.out / "fig2d_descriptor_ranking.png")
    if command == "discover":
        done["discover"] = knowledge.discover(lit, args.rounds, max(16, args.starts // 4),
                                              importance=ranking.rank(lit)["importance"])
    if command in ("fit", "run"):
        done["fit"] = report.fit_all(lit, exp, args.starts)
        report.figures(done["fit"], lit, exp, args.out)
    if command in ("recommend", "run"):
        done["recommend"], _ = action.recommend(lit, exp, args.batch, starts=max(16, args.starts // 4))
    (args.out / f"{args.command}.json").write_text(json.dumps(done, indent=2, ensure_ascii=False), encoding="utf-8")
    print(report.summary(done))


if __name__ == "__main__":
    main()
