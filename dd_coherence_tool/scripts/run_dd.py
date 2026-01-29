#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Assure l'import local sans installation
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from dd_coherence import DDParams, run_dd_coherence, write_outputs  # noqa: E402


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


def main():
    ap = argparse.ArgumentParser(description="DD Cohérence IA (diagnostic descriptif de transitions de phase)")
    ap.add_argument("--input", required=True, help="CSV d'entrée")
    ap.add_argument("--outdir", required=True, help="Dossier de sortie")
    ap.add_argument("--time-col", default=None, help="Nom de colonne temps (optionnel)")
    ap.add_argument("--u-col", default=None, help="Nom de colonne exogène u_t (optionnel)")
    ap.add_argument("--cols", default=None, help="Liste de colonnes séparées par des virgules. Défaut: toutes les colonnes numériques hors time/u.")
    ap.add_argument("--nan-policy", default="zero", choices=["zero", "ffill", "drop", "none"], help="Gestion NaN pour X et u")
    ap.add_argument("--config", default=None, help="Fichier JSON de paramètres (ex ante)")
    ap.add_argument("--no-components", action="store_true", help="Ne pas écrire dd_components.csv.gz")
    args = ap.parse_args()

    df = pd.read_csv(args.input)

    params = load_params(args.config)

    if args.cols:
        cols = [c.strip() for c in args.cols.split(",") if c.strip()]
    else:
        exclude = set()
        if args.time_col:
            exclude.add(args.time_col)
        if args.u_col:
            exclude.add(args.u_col)
        cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    if not cols:
        raise SystemExit("Aucune colonne numérique sélectionnée pour DD.")

    result = run_dd_coherence(
        df=df,
        cols=cols,
        params=params,
        time_col=args.time_col,
        u_col=args.u_col,
        nan_policy=args.nan_policy,
    )

    outputs = write_outputs(
        result=result,
        out_dir=args.outdir,
        input_path=args.input,
        code_root=str(ROOT),
        write_components_csv_gz=not args.no_components,
    )

    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
