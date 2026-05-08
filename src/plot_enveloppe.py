from pde_residuals import compute_stresses
import torch
import pytest
import numpy as np
from layers import NeuralNet
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt

def plot_operational_envelope(L, H, K_scale, device):
    """
    Génère la courbe d'erreur sur l'enveloppe complète.
    Sweeps indépendants sur E, P, nu (les 2 autres fixés à leur valeur médiane).
    """
    
    E_median, P_median, nu_median = 87.5e9, 2750, 0.275
    I = H**3 / 12



    model = NeuralNet(hidden_dim=100, use_fourier=True).to(device)
    try:
        state_dict = torch.load("A2S2_model_V0_30.pth", map_location=device)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        pytest.skip("Modèle A2S2_model_V0_30.pth non trouvé.")
    
    model.eval()
    
    # 3. Inférence au bout de la poutre (x=L, y=0)
    x_test = torch.tensor([[L]], device=device, requires_grad=True)
    y_test = torch.tensor([[0.0]], device=device, requires_grad=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Enveloppe Opérationnelle A2S2 V0.30")

    # --- Sweep E (P et nu fixés) ---
    E_range = np.linspace(10e9, 200e9, 50)
    errors_E = []
    for E_val in E_range:

        # 1. Calculer v_max_theory avec E_val, P_median
        v_max_theory = (P_median * L**3) / (3 * E_val * I)

        # 2. Préparer tenseurs (E_adim, p_adim fixe, nu_t fixe)
        E_adim = E_val/140e9
        p_adim = (P_median*K_scale)/140e9
        E = torch.full_like(x_test, E_adim)
        p = torch.full_like(x_test, p_adim)
        nu_t = torch.full_like(x_test, nu_median)

        # 3. Inférence -> v_pinn
        
        u, v = model(x_test, y_test, E, nu_t, p)
        u_pred = u / K_scale
        v_pred = v / K_scale
        
        v_max_pinn = abs(v_pred[0, 0].item()) # On prend la valeur absolue de la flèche mise à l'échelle

        # 4. Calcul de l'erreur
        error_relative = abs(v_max_pinn - v_max_theory) / v_max_theory
        errors_E.append(error_relative*100)

        pass

    axes[0].plot(E_range / 1e9, errors_E)
    axes[0].axhline(y=1, color='r', linestyle='--', label='Seuil 1%')
    axes[0].axvspan(35, 140, alpha=0.1, color='green', label='Domaine entraîn.')
    axes[0].set(title='Sensibilité à E', xlabel='E (GPa)', ylabel='Erreur (%)')

    # --- Sweep P (même logique) ---
    P_range = np.linspace(200, 8000, 50)
    errors_P = []
    for P_val in P_range:

        # 1. Calculer v_max_theory
        v_max_theory = (P_val * L**3) / (3 * E_median * I)

        # 2. Préparer tenseurs 
        E_adim = E_median/140e9
        p_adim = (P_val*K_scale)/140e9

        E = torch.full_like(x_test, E_adim)
        p = torch.full_like(x_test, p_adim)
        nu_t = torch.full_like(x_test, nu_median)

        # 3. Inférence -> v_pinn
        
        u, v = model(x_test, y_test, E, nu_t, p)
        u_pred = u / K_scale
        v_pred = v / K_scale
        
        v_max_pinn = abs(v_pred[0, 0].item()) # On prend la valeur absolue de la flèche mise à l'échelle

        # 4. Calcul de l'erreur
        error_relative = abs(v_max_pinn - v_max_theory) / v_max_theory
        errors_P.append(error_relative*100)
        pass
    
    axes[1].plot(P_range, errors_P)
    axes[1].axhline(y=1, color='r', linestyle='--', label='Seuil 1%')
    axes[1].axvspan(500, 5000, alpha=0.1, color='green', label='Domaine entraîn.')
    axes[1].set(title='Sensibilité à P', xlabel='P (N)', ylabel='Erreur (%)')

    # => Le sweep nu test si le réseau est stable sur nu, pas si nu change la physique 1D
    nu_range = np.linspace(0.05, 0.6, 50)
    errors_nu = []
    for nu_val in nu_range:

        # 1. Calculer v_max_theory
        v_max_theory = (P_median * L**3) / (3 * E_median * I)

        # 2. Préparer tenseurs
        E_adim = E_median/140e9
        p_adim = (P_median*K_scale)/140e9
        E = torch.full_like(x_test, E_adim)
        p = torch.full_like(x_test, p_adim)
        nu_t = torch.full_like(x_test, nu_val)

        # 3. Inférence -> v_pinn
        
        u, v = model(x_test, y_test, E, nu_t, p)
        u_pred = u / K_scale
        v_pred = v / K_scale
        
        v_max_pinn = abs(v_pred[0, 0].item()) # On prend la valeur absolue de la flèche mise à l'échelle

        # 4. Calcul de l'erreur
        error_relative = abs(v_max_pinn - v_max_theory) / v_max_theory
        errors_nu.append(error_relative*100)
        pass
    
    axes[2].plot(nu_range, errors_nu)
    axes[2].axhline(y=1, color='r', linestyle='--', label='Seuil 1%')
    axes[2].axvspan(0.1, 0.45, alpha=0.1, color='green', label='Domaine entraîn.')
    axes[2].set(title='Sensibilité à nu', xlabel='nu', ylabel='Erreur (%)')

    plt.tight_layout()
    plt.savefig("Envelope_A2S2_V0_30_test.png", dpi=150)


if __name__ == "__main__" : 
    plot_operational_envelope(L=1.0, H=0.1, K_scale=1e5, device="cpu")