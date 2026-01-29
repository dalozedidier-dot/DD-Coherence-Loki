# DD Cohérence IA (DD) – Package GitHub complet

Ce dossier est prêt à être copié tel quel dans un dépôt GitHub.

## Contenu
- dd_coherence_tool/ : le moteur DD + CLI + tests
- .github/workflows/dd_smoke.yml : CI GitHub Actions (smoke + artefacts)
- .gitignore

## Utilisation locale
### 1) Installer les dépendances
```bash
python -m pip install --upgrade pip
pip install -r dd_coherence_tool/requirements.txt
```

### 2) Lancer sur un CSV
```bash
python dd_coherence_tool/scripts/run_dd.py --input data.csv --outdir out_dd --cols x
```

Si ton CSV a une colonne temps, tu peux préciser :
```bash
python dd_coherence_tool/scripts/run_dd.py --input data.csv --outdir out_dd --time-col date --cols x
```

Optionnel : signal exogène u_t
```bash
python dd_coherence_tool/scripts/run_dd.py --input data.csv --outdir out_dd --cols x --u-col u
```

## Sorties
Le dossier de sortie contient notamment :
- dd_report.json : rapport complet (transitions, séries, paramètres, hashes)
- dd_series.csv : t, Phi, F, Cand
- dd_components.csv.gz : détails des composantes (si activé)

## CI GitHub
Dès que tu pousses ce dossier dans un repo, le workflow `dd-smoke` tourne sur push et pull_request.
Il génère un CSV synthétique, exécute DD, puis publie les fichiers de sortie comme artefacts.

## Remarque
Le code est autonome, mais il faut Python et les dépendances installées via pip.
