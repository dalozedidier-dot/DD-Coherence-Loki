# Fichiers Git ajoutés pour exécuter DD sur 1.csv, 2.csv, 3.csv

## Où copier ces fichiers
Copie à la racine du repo :
- dd_params.small.json
- .github/workflows/dd_on_committed_csvs.yml

Copie dans dd_coherence_tool/scripts :
- run_dd_batch.py

Les fichiers 1.csv, 2.csv, 3.csv doivent être dans dd_coherence_tool/ (comme tu l'as fait).

## Ce que fait la CI
Le workflow lance un batch :
- out = _ci_out/dd_runs/<nom_du_csv_sans_extension>/
- artefacts uploadés : dd_report.json, dd_series.csv, dd_components.csv.gz si possible
- si un fichier est trop court (ex: 2.csv avec 1 ligne), il est marqué skipped et un dd_skipped.json est écrit

## Run local
```bash
pip install -r dd_coherence_tool/requirements.txt
python dd_coherence_tool/scripts/run_dd_batch.py --outdir out_dd --config dd_params.small.json dd_coherence_tool/1.csv dd_coherence_tool/2.csv dd_coherence_tool/3.csv
```
