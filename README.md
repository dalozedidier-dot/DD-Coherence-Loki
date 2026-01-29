# DD Cohérence – Repo prêt à pousser

Ce dossier est un dépôt complet. Tu peux le pousser tel quel sur GitHub.

## Ce que tu as ici
1) dd_coherence_tool/ : moteur DD + scripts + tests + tes CSV (1.csv, 2.csv, 3.csv)
2) .github/workflows/
   - dd_smoke.yml : smoke test sur un CSV synthétique, produit aussi dd_components.csv.gz
   - dd_on_committed_csvs.yml : exécute DD sur dd_coherence_tool/1.csv, 2.csv, 3.csv

## Important sur tes CSV actuels
Les fichiers 1.csv, 2.csv, 3.csv que tu as déposés ici sont des tableaux de synthèse (1 ligne pour certains).
DD nécessite une longueur minimale. Un CSV avec 1 ligne sera automatiquement marqué "skipped".

Pour que DD ait du sens, il faut une série (plusieurs points) et un ordre (temps ou séquence).

## Lancer en local
```bash
pip install -r dd_coherence_tool/requirements.txt
python dd_coherence_tool/scripts/run_dd.py --input dd_coherence_tool/1.csv --outdir out_dd --cols "'Failure rate" "'Avg job run time"
```

Batch (sur 1.csv 2.csv 3.csv) :
```bash
python dd_coherence_tool/scripts/run_dd_batch.py --outdir out_dd --config dd_params.small.json dd_coherence_tool/1.csv dd_coherence_tool/2.csv dd_coherence_tool/3.csv
```

## Sorties
Chaque run écrit :
- dd_report.json
- dd_series.csv
- dd_components.csv.gz (si non désactivé et si la série est assez longue)

## GitHub Actions
Une fois pushé, va dans Actions, ouvre un run, puis récupère l'artefact "dd_outputs" ou "dd_runs".
