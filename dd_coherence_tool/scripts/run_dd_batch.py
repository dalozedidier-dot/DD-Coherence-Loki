#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, List

# Assure l'import local sans installation
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dd_coherence import DDParams, run_dd_coherence, write_outputs  # noqa: E402


def _sanitize_name(s: str) -> str:
    # Nettoie les colonnes du type '"\'Avg run time"' en "Avg run time"
    s2 = s.strip()
    s2 = re.sub(r"^[\s\"'\\]+", "", s2)
    s2 = re.sub(r"[\s\"'\\]+$", "", s2)
    return s2.strip()


def sanitize_df(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, str]]:
    mapping: Dict[str, str] = {}
    new_cols: List[str] = []
    for c in df.columns:
        nc = _sanitize_name(str(c))
        if nc in mapping.values():
            # évite collisions: ajoute suffixe
            base = nc
            k = 2
            while f"{base}_{k}" in mapping.values():
                k += 1
            nc = f"{base}_{k}"
        mapping[str(c)] = nc
        new_cols.append(nc)
    out = df.copy()
    out.columns = new_cols
    return out, mapping


def load_params(config_path: str | None) -> DDParams:
    if not config_path:
        return DDParams()
    p = Path(config_path)
    obj = json.loads(p.read_text(encoding="utf-8"))
    base = DDParams().__dict__.copy()
    for k, v in obj.items():
        if k not in base:
            raise ValueError(f"Paramètre inconnu dans config: {k}")
        base[k] = v
    return DDParams(**base)


def fit_params_to_length(params: DDParams, n: int) -> DDParams | None:
    # Ajuste m, k, r pour éviter l'erreur "série trop courte".
    # Si n est trop petit, renvoie None (skip).
    if n < 5:
        return None

    m = min(params.m, n)
    m = max(3, m)

    # On veut 2k + r + 1 <= n. On réduit k puis r si nécessaire.
    k = min(params.k, max(1, (n - 2) // 3))
    r = min(params.r, max(1, (n - 2) // 6))

    # Ajustements fins pour respecter la contrainte
    while 2 * k + r + 1 > n and k > 1:
        k -= 1
    while 2 * k + r + 1 > n and r > 1:
        r -= 1

    if 2 * k + r + 1 > n:
        return None

    base = params.__dict__.copy()
    base["m"] = int(m)
    base["k"] = int(k)
    base["r"] = int(r)
    return DDParams(**base)


def select_numeric_cols(df: pd.DataFrame, exclude: set[str]) -> List[str]:
    cols: List[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    if cols:
        return cols

    # fallback: tentative de coercition
    for c in df.columns:
        if c in exclude:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        ok = np.isfinite(s.to_numpy(dtype=float)).mean()
        if ok >= 0.8 and s.nunique(dropna=True) >= 3:
            df[c] = s
            cols.append(c)

    return cols


def main():
    ap = argparse.ArgumentParser(description="Batch runner DD Cohérence IA sur plusieurs CSV")
    ap.add_argument("--outdir", required=True, help="Dossier de sortie racine")
    ap.add_argument("--config", default=None, help="JSON paramètres ex ante (optionnel)")
    ap.add_argument("--time-col", default=None, help="Colonne temps (optionnel)")
    ap.add_argument("--u-col", default=None, help="Colonne exogène u_t (optionnel)")
    ap.add_argument("--nan-policy", default="zero", choices=["zero", "ffill", "drop", "none"], help="Gestion NaN")
    ap.add_argument("--keep-original-columns", action="store_true", help="Ne pas nettoyer les noms de colonnes")
    ap.add_argument("inputs", nargs="+", help="Chemins CSV à traiter")
    args = ap.parse_args()

    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    base_params = load_params(args.config)

    summary: Dict[str, Any] = {"runs": []}

    for in_path in args.inputs:
        p = Path(in_path)
        run_entry: Dict[str, Any] = {"input": str(p), "status": "ok"}

        try:
            df = pd.read_csv(p)
        except Exception as e:
            run_entry["status"] = "error"
            run_entry["error"] = f"read_csv: {e}"
            summary["runs"].append(run_entry)
            continue

        colmap = None
        if not args.keep_original_columns:
            df, colmap = sanitize_df(df)
            run_entry["column_map"] = colmap

        N = len(df)
        fitted = fit_params_to_length(base_params, N)
        if fitted is None:
            run_entry["status"] = "skipped"
            run_entry["reason"] = f"série trop courte (N={N}) pour calculer des fenêtres"
            # écrit un petit fichier de trace
            sk_dir = out_root / p.stem
            sk_dir.mkdir(parents=True, exist_ok=True)
            (sk_dir / "dd_skipped.json").write_text(json.dumps(run_entry, ensure_ascii=False, indent=2), encoding="utf-8")
            summary["runs"].append(run_entry)
            continue

        exclude = set()
        if args.time_col:
            exclude.add(args.time_col)
        if args.u_col:
            exclude.add(args.u_col)

        cols = select_numeric_cols(df, exclude)
        if not cols:
            run_entry["status"] = "skipped"
            run_entry["reason"] = "aucune colonne numérique détectée"
            sk_dir = out_root / p.stem
            sk_dir.mkdir(parents=True, exist_ok=True)
            (sk_dir / "dd_skipped.json").write_text(json.dumps(run_entry, ensure_ascii=False, indent=2), encoding="utf-8")
            summary["runs"].append(run_entry)
            continue

        run_entry["used_cols"] = cols
        run_entry["used_params"] = fitted.__dict__.copy()

        try:
            result = run_dd_coherence(
                df=df,
                cols=cols,
                params=fitted,
                time_col=args.time_col,
                u_col=args.u_col,
                nan_policy=args.nan_policy,
            )

            out_dir = out_root / p.stem
            outputs = write_outputs(
                result=result,
                out_dir=str(out_dir),
                input_path=str(p),
                code_root=str(ROOT),
                write_components_csv_gz=True,
            )
            run_entry["outputs"] = outputs

        except Exception as e:
            run_entry["status"] = "error"
            run_entry["error"] = str(e)
            err_dir = out_root / p.stem
            err_dir.mkdir(parents=True, exist_ok=True)
            (err_dir / "dd_error.json").write_text(json.dumps(run_entry, ensure_ascii=False, indent=2), encoding="utf-8")

        summary["runs"].append(run_entry)

    (out_root / "dd_batch_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
