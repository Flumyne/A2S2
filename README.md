# A2S2 Core : Aero-AI-Space-Solver

Moteur de simulation **Physics-ML "Mesh-Free"** basé sur la **Deep Energy Method (DEM)**, codé entièrement à la main (hand-coded). Conçu pour le New Space : remplacer les simulations FEA lourdes par des inférences ultra-rapides (<100ms).

> **Dernière validation** : Poutre encastrée 2D (Aluminium 7075-T6) — Erreur **2.48%** vs. théorie d'Euler-Bernoulli. ✅

## Structure du Code

| Fichier | Rôle |
|---------|------|
| `layers.py` | Architecture MLP (Tanh, tête duale u/v, Normalizer) |
| `pde_residuals.py` | Physique hand-coded : déformations, Hooke, `compute_strain_energy` |
| `solver.py` | Boucle DEM : `Loss = StrainEnergy - ExtWork + λ·DirichletBC` |
| `data_gen.py` | Génération de nuages de points (collocation, bords) |
| `post_process.py` | Inférence, calcul de Von Mises, visualisation |
| `utils.py` | Normalizer, `visualize_loss` |

## Philosophie "Hand-Coded"
1. **Zéro boîte noire** : Pas de Modulus, pas de DeepXDE. Chaque gradient est calculé et compris.
2. **Autograd pur** : `torch.autograd.grad` avec `create_graph=True` pour les dérivées des dérivées.
3. **Variationnel** : La loss est une énergie physique, pas une somme de résidus — jamais de Shear Locking.

