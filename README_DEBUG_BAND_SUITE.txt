But
- Obtenir des logs complets et un artefact meme quand band_suite echoue.
- Eviter le blocage "Entrypoint introuvable" en te laissant fournir la commande exacte.

A faire dans GitHub (optionnel mais recommande)
Settings -> Secrets and variables -> Actions -> Variables
Ajouter:
ACTIONS_STEP_DEBUG = true
ACTIONS_RUNNER_DEBUG = true

Utilisation
Actions -> band-suite (isolated) -> Run workflow
- debug = true
- band_cmd = commande exacte si tu la connais

Exemples band_cmd
- python scripts/ci_band_suite.py
- pytest -q tests/band_suite
- python -m ton_module

Resultat
- Un artefact band_suite_artifacts sera toujours uploade.
- _ci_out/find_band_suite.txt liste les candidats.
- Logs: _ci_out/band_suite_attempt_1.log, etc.
