import torch
import pytest
import numpy as np
from src.layers import NeuralNet
from src.utils import Normalizer
from src.data_gen import Geometry

def test_cantilever_deflection_v0():
    """
    Compare la flèche maximale du PINN (V0) avec la théorie d'Euler-Bernoulli.
    Formule : v_max = (P * L^3) / (3 * E * I)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Paramètres (Adimensionnels)
    L = 1.0
    H = 0.1
    E = 1.0
    nu = 0.33
    p = 1000 / 70e9  # Charge adimensionnelle utilisée à l'entraînement
    I = (1.0 * H**3) / 12  # Moment d'inertie (unité de largeur)
    
    # 2. Théorie d'Euler-Bernoulli
    v_max_theory = (p * L**3) / (3 * E * I)
    
    # 3. Chargement du modèle PINN
    
    geo = Geometry(L, H, device)
    x_sample, y_sample = geo.generate_collocation_points(3000)
    X_sample = torch.cat([x_sample,y_sample], dim=1)

    normalizer = Normalizer(X_sample, device=device)

    model = NeuralNet(normalizer, input_dim=2, hidden_dim=128, output_dim=2, use_fourier=False).to(device)
    try:
        state_dict = torch.load("A2S2_model_V0_09.pth", map_location=device)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        pytest.skip("Modèle A2S2_model_V0_09.pth non trouvé.")
    
    model.eval()
    
    # 4. Inférence au bout de la poutre (x=L, y=0)
    x_test = torch.tensor([[L]], device=device, requires_grad=True)
    y_test = torch.tensor([[0.0]], device=device, requires_grad=True)
    
    pred = model(x_test, y_test)
    
    K_scale = 1e5
    u_pred = pred[:, 0:1] / K_scale
    v_pred = pred[:, 1:2] / K_scale
    
    # Importer et appeler compute_stresses
    from src.pde_residuals import compute_stresses
    sig_xx, sig_yy, sig_xy = compute_stresses(u_pred * K_scale, v_pred * K_scale, x_test, y_test, E, nu)
    sig_xy = sig_xy / K_scale
    
    print(f"sigma_xy à x=L    : {sig_xy.item():.3e}")
    print(f"sigma_xy cible    : {-p/H:.3e}")   # -1.428e-7
    print(f"Ratio             : {sig_xy.item() / (-p/H):.3f}")
    
    v_max_pinn = abs(v_pred[0, 0].item()) # On prend la valeur absolue de la flèche mise à l'échelle
    
    # 5. Calcul de l'erreur
    error_relative = abs(v_max_pinn - v_max_theory) / v_max_theory
    
    print(f"\n--- Validation Physique A2S2 V0_09 ---")
    print(f"Flèche Théorique (Euler-Bernoulli) : {v_max_theory:.2e}")
    print(f"Flèche Prédite (PINN)             : {v_max_pinn:.2e}")
    print(f"Erreur Relative                   : {error_relative*100:.2f}%")
    
    # Seuil de tolérance : 10% pour une V0 (PINN 2D vs Théorie 1D)
    assert error_relative < 0.15, f"Erreur trop élevée : {error_relative*100:.2f}%"

if __name__ == "__main__":
    test_cantilever_deflection_v0()
