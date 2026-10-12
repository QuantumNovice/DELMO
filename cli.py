"""``delmo`` command line (or ``python -m delmo``).

    delmo info
    delmo predict model.ifc [--out predictions.csv] [--apply model.corrected.ifc] [--tau 0.95]
    delmo renders <folder of GlobalId-IfcClass-view.png> [--ifc source.ifc] [--out predictions.csv]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="delmo", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None, help="checkpoint folder (default: the released one)")
    ap.add_argument("--device", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info", help="model, training data and test results of the checkpoint")
    p = sub.add_parser("predict", help="classify and route every entity of an IFC file")
    p.add_argument("ifc")
    p.add_argument("--out", default=None)
    p.add_argument("--apply", default=None, help="write a corrected IFC with the auto decisions")
    p.add_argument("--tau", type=float, default=None)
    p.add_argument("--limit", type=int, default=None)
    p = sub.add_parser("renders", help="classify a folder of rendered views (dataset layout)")
    p.add_argument("folder")
    p.add_argument("--ifc", default=None, help="source IFC for the neighbourhoods")
    p.add_argument("--out", default="delmo_predictions.csv")
    p.add_argument("--tau", type=float, default=None)
    a = ap.parse_args(argv)

    from .predictor import CHECKPOINT

    ck = Path(a.checkpoint or CHECKPOINT)
    if a.cmd == "info":
        print((ck / "config.json").read_text(encoding="utf-8"))
        return 0
    if a.cmd in ("predict", "renders"):
        from .predictor import Predictor, summary, write_corrected

        pr = Predictor(ck, a.device)
        tau = pr.tau if a.tau is None else a.tau
        if a.cmd == "predict":
            df = pr.predict_ifc(a.ifc, tau, limit=a.limit)
            out = Path(a.out or Path(a.ifc).with_suffix(".delmo.csv"))
        else:
            df = pr.predict_renders(a.folder, a.ifc, tau)
            out = Path(a.out)
        df.to_csv(out, index=False)
        s = summary(df, tau)
        out.with_suffix(".summary.json").write_text(json.dumps(s, indent=2))
        print(json.dumps(s, indent=2))
        if a.cmd == "predict" and a.apply and len(df):
            rows = write_corrected(a.ifc, df, a.apply)
            print(f"{sum(r['status'] == 'ok' for r in rows)} entities corrected -> {a.apply}")
            df[df.decision == "refer"].to_csv(Path(a.apply).with_suffix(".referrals.csv"), index=False)
        print(f"predictions -> {out}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
