# A2S2 Core : Aero-AI-Space-Solver

Moteur de simulation **Physics-ML "Mesh-Free"** basé sur la **Deep Energy Method (DEM)** paramétrique, codé entièrement à la main (hand-coded). Conçu pour le New Space : remplacer les simulations FEA lourdes par des inférences ultra-rapides (<100ms) capables de généraliser sur un domaine opérationnel.

> **État Actuel (V0.30) - Méta-Solveur FSI Structure** : Solveur paramétrique (CPDEM) avec Biais Inductif Physique. Erreur validée à **< 1%** sur l'enveloppe opérationnelle complète ($E, \nu, P$). ✅

## Milestones & Architecture

*   **CPDEM (Continuous Parametric Deep Energy Method)** : Le solveur n'apprend pas qu'une géométrie, mais la physique continue sur un espace de paramètres ($E \in [10, 200]$ GPa, $P \in [200, 8000]$ N).
*   **Biais Inductif Physique** : Mise à l'échelle analytique proportionnelle à $P/E$ dans l'architecture `NeuralNet`.
*   **Fourier Features** : Cartographie des entrées spatiales pour contrer le biais spectral des MLPs et capturer les hautes fréquences de déformation.

## Structure du Code

| Fichier | Rôle |
|---------|------|
| `layers.py` | Architecture `NeuralNet` (Fourier, SiLU, biais physique $P/E$) |
| `pde_residuals.py` | Physique hand-coded : contraintes, `compute_strain_energy` |
| `solver.py` | Boucle DEM paramétrique : `Loss = StrainEnergy - ExtWork + λ·DirichletBC` |
| `data_gen.py` | Génération de points de collocation spatiaux et paramétriques (Log-Sampling) |
| `plot_enveloppe.py` | Test d'inférence en balayage sur l'enveloppe opérationnelle |
| `post_process.py` | Inférence, calcul de Von Mises, visualisation statique |

## Philosophie "Hand-Coded"
1. **Zéro boîte noire** : Chaque gradient PDE est calculé via `torch.autograd.grad`.
2. **Variationnel (DEM)** : Optimisation de l'énergie potentielle (Ritz-Galerkin via NN) évitant le *Shear Locking* naturel des PINNs de collocation forts.
3. **Startup-Ready** : Rigueur documentaire et visualisation ciblée pour le portfolio.
