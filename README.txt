Correctif band_suite

Constat (depuis band_suite_artifacts.zip)
- Aucun script band_suite n'existait dans le repo. Donc le workflow ne pouvait pas tourner.

Ce pack ajoute:
- scripts/ci_band_suite.py : un entrypoint réel
  - exécute les tests DD si présents
  - lance un smoke DD qui produit dd_components.csv.gz
  - optionnel: lance run_dd_batch.py si présent
  - écrit des logs dans _ci_out/

- .github/workflows/band_suite_isolated.yml : workflow qui appelle scripts/ci_band_suite.py et uploade _ci_out

À faire
1) Copie scripts/ci_band_suite.py dans ton repo (crée le dossier scripts/ s'il n'existe pas)
2) Remplace .github/workflows/band_suite_isolated.yml par la version fournie
3) Commit et push

Résultat
- band-suite (isolated) devient vert et produit toujours un artefact _ci_out exploitable.
