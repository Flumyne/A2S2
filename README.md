# A2S2 Core : Aero-AI-Space-Solver Logic

Ce dossier contient le cœur algorithmique du solveur **A2S2**, développé entièrement à la main (hand-coded) pour une maîtrise totale des gradients et de la physique.

## Structure du Code
- `layers.py` : Implémentation manuelle des couches de neurones, fonctions d'activation et initialisations.
- `pde_residuals.py` : Définition des résidus des équations aux dérivées partielles (Navier-Stokes, Élasticité) et conditions aux limites.
- `solver.py` : Boucle d'optimisation, gestion du couplage FSI et intégration temporelle.
- `utils.py` : Fonctions d'aide (visualisation, export de données, métriques de convergence).

## Philosophie "Hand-Coded"
1. Pas de frameworks de haut niveau (Modulus, DeepXDE) pour le cœur de calcul.
2. Utilisation de la différenciation automatique (Autograd) ou calcul manuel des Jacobiennes pour le couplage fort.
3. Transparence totale sur le calcul de la fonction de perte (Loss).
