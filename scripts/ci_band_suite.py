#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
from pathlib import Path
import json
import datetime


def sh(cmd: list[str], *, cwd: Path | None = None, out_log: Path | None = None) -> int:
    if out_log:
        out_log.parent.mkdir(parents=True, exist_ok=True)
        with out_log.open("w", encoding="utf-8") as f:
            f.write("+ " + " ".join(cmd) + "\n")
            f.flush()
            p = subprocess.Popen(cmd, cwd=str(cwd) if cwd else None, stdout=f, stderr=subprocess.STDOUT, text=True)
            return p.wait()
    else:
        print("+", " ".join(cmd))
        return subprocess.call(cmd, cwd=str(cwd) if cwd else None)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    ci_out = root / "_ci_out"
    ci_out.mkdir(parents=True, exist_ok=True)

    meta = {
        "utc": datetime.datetime.utcnow().isoformat() + "Z",
        "pwd": str(root),
        "python": subprocess.check_output(["python", "-V"], text=True).strip(),
        "pip": subprocess.check_output(["pip", "-V"], text=True).strip(),
    }
    (ci_out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # 1) Tests DD si présents
    dd_tests = root / "dd_coherence_tool" / "tests"
    if dd_tests.exists():
        rc = sh(["python", "-m", "pytest", "-q", str(dd_tests)], out_log=ci_out / "pytest_dd.log")
        if rc != 0:
            return rc

    # 2) Smoke DD avec composants (génère dd_report.json, dd_series.csv, dd_components.csv.gz)
    sample_csv = ci_out / "sample.csv"
    out_dd = ci_out / "out_dd"
    out_dd.mkdir(parents=True, exist_ok=True)

    # Génère un CSV synthétique
    rc = sh(["python", "-c",
             "import numpy as np, pandas as pd; "
             "np.random.seed(0); "
             "x=np.r_[np.random.normal(0,1,200),np.random.normal(3,1.5,200)]; "
             "y=np.r_[np.random.normal(0,1,200),np.random.normal(0,1,200)]; "
             "pd.DataFrame({'x':x,'y':y}).to_csv(r'%s', index=False)"
             % str(sample_csv)],
            out_log=ci_out / "make_sample.log")
    if rc != 0:
        return rc

    run_dd = root / "dd_coherence_tool" / "scripts" / "run_dd.py"
    if run_dd.exists():
        rc = sh(["python", str(run_dd),
                 "--input", str(sample_csv),
                 "--outdir", str(out_dd),
                 "--cols", "x,y"],
                out_log=ci_out / "run_dd_smoke.log")
        if rc != 0:
            return rc

    # 3) Batch sur CSV commités si le runner existe
    run_batch = root / "dd_coherence_tool" / "scripts" / "run_dd_batch.py"
    if run_batch.exists():
        csvs = sorted([p for p in (root / "dd_coherence_tool").glob("*.csv")
                      if not p.name.endswith("_uuid.csv") and not p.name.endswith("_orig.csv")])
        if csvs:
            cmd = ["python", str(run_batch),
                   "--outdir", str(ci_out / "dd_runs"),
                   "--config", str(root / "dd_params.small.json")] + [str(p) for p in csvs]
            rc = sh(cmd, out_log=ci_out / "run_dd_batch.log")
            if rc != 0:
                return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
