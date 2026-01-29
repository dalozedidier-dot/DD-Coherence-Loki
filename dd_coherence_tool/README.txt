DD Cohérence IA
Outil strictement descriptif de diagnostic de transitions de phase (framework DD).

Ce dossier implémente uniquement DD selon la formule "D.D Cohérence IA".
Aucune logique DD-R et aucune logique Équilibre E n'est incluse.

Entrée attendue
- Un CSV avec une colonne temps optionnelle (time-col)
- Une ou plusieurs colonnes numériques (cols) représentant x_t ∈ R^d
- Optionnel: une colonne u_t exogène (u-col)

Sorties
- dd_report.json
  - instants {t_k*}
  - séries Φ_t, F_t, Cand(t)
  - snapshots de métriques locales associées aux transitions
  - paramètres exacts
  - hash code + données
- dd_series.csv
  - t, Phi, F, Cand
- dd_components.csv.gz (optionnel)
  - long format sur transitions: C_tj, delta_tj et mesures brutes (V,A,CHI,H)

Exécution
1) Dépendances:
   pip install numpy pandas scipy

2) Lancer:
   python scripts/run_dd.py --input data.csv --outdir out_dd --time-col date --u-col u

3) Paramètres ex ante via JSON:
   python scripts/run_dd.py --input data.csv --outdir out_dd --config dd_params.json

Exemple dd_params.json:
{
  "m": 30,
  "k": 10,
  "r": 5,
  "cooldown": 20,
  "K": 20,
  "N0": 200,
  "Lref": 200,
  "a": 0,
  "zmax": 6.0,
  "q": 0.2,
  "theta": 0.75,
  "p": 0.3,
  "p_C": 0.3,
  "kappa": 0.10,
  "kappa_C": 0.10
}

Notes
- Le choix des bins d'entropie est fixé à partir de la référence initiale T0^(0) = [m-1, N0].
- Si aucun u_t n'est fourni, CHI est fixé à 0 (comme dans la formule).
- L'absence de transition t* est un résultat valide.

Intégration GitHub Actions
- Ajoute ce dossier dans un dépôt GitHub.
- Ajoute le workflow: .github/workflows/dd_smoke.yml
- Le workflow installe les dépendances, lance pytest, exécute un run DD sur un CSV synthétique et uploade les artefacts.

GitHub Actions (prêt à l'emploi)
- Un workflow CI est fourni dans .github/workflows/dd_smoke.yml.
- Il installe les dépendances, exécute le smoke test, génère un CSV synthétique, lance DD, et publie les artefacts.

Repo minimal
- Racine du repo
  - dd_coherence_tool/
  - .github/workflows/dd_smoke.yml
  - .gitignore
