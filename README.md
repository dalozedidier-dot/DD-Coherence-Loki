# DD Cohérence – Repo prêt à pousser

Ce dossier est un dépôt complet. Tu peux le pousser tel quel sur GitHub.

## Ce que tu as ici
1) dd_coherence_tool/ : moteur DD + scripts + tests + tes CSV (1.csv, 2.csv, 3.csv)
2) .github/workflows/
   - dd_smoke.yml : smoke test sur un CSV synthétique, produit aussi dd_components.csv.gz
   - dd_on_committed_csvs.yml : exécute DD sur dd_coherence_tool/1.csv, 2.csv, 3.csv

## Données de test (non-skipped)
Les fichiers `dd_coherence_tool/1.csv`, `2.csv`, `3.csv` contiennent désormais des **séries temporelles synthétiques** (N≥120) avec **ruptures de régime** afin de produire des sorties DD **non-skipped** en local et en CI.

- Les anciennes versions "tableau de synthèse / trop courtes" sont conservées sous `*_orig.csv`.
- Les variantes `*_uuid.csv` restent disponibles pour tests de renommage/identifiants.

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
