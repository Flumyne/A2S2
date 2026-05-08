from src.pde_residuals import compute_stresses
import torch
import pytest
import numpy as np
from src.layers import NeuralNet
from src.utils import Normalizer
from src.data_gen import Geometry




def test_parametric_sweep():
    """
    #Compare la flèche maximale du PINN avec la théorie d'Euler-Bernoulli sur différents emplacement des plages paramètrique
    #Formule : v_max = (P * L^3) / (3 * E * I)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Paramètres (Adimensionnels)
    L = 1.0
    H = 0.1

    test_cases = [
        {"E": 35e9, "nu":0.1, "P": 500 }, # Bord "Mou"
        {"E": 140e9, "nu":0.45, "P": 5000}, # Bord Rigide
        {"E": 87.5e9, "nu":0.275, "P": 2750}, # Median
    ]

    # LOAD SCALING : On multiplie la force par 100 000 pour éviter l'underflow Float32
    K_scale = 1e5

    # 2. Chargement du modèle PINN
    
    geo = Geometry(L, H, device)
    x_sample, y_sample = geo.generate_collocation_points(5000)
    E_ref_sample = (140e9 - 35e9)*torch.rand(5000,1).to(device) + 35e9
    nu_sample = (0.45 - 0.1)*torch.rand(5000,1).to(device) + 0.1
    p_sample = (5000 - 500)*torch.rand(5000,1).to(device) + 500
    E_sample = E_ref_sample/140e9
    p_sample = (p_sample * K_scale) / 140e9

    torch.manual_seed(42)
    X_sample_spatial = torch.cat([x_sample,y_sample], dim=1)
    X_sample_param = torch.cat([E_sample, nu_sample, p_sample], dim=1)

    normalizer_spatial = Normalizer(X_sample_spatial, device=device)
    normalizer_param = Normalizer(X_sample_param, device=device)

    model = NeuralNet(normalizer_spatial, normalizer_param, hidden_dim=100, use_fourier=True).to(device)
    try:
        state_dict = torch.load("A2S2_model_V0_298.pth", map_location=device)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        pytest.skip("Modèle A2S2_model_V0_298.pth non trouvé.")
    
    model.eval()
    
    # 3. Inférence au bout de la poutre (x=L, y=0)
    x_test = torch.tensor([[L]], device=device, requires_grad=True)
    y_test = torch.tensor([[0.0]], device=device, requires_grad=True)

    for case in test_cases:

        E = case["E"]
        nu = case["nu"]
        I = (1.0 * H**3) / 12  # Moment d'inertie (unité de largeur)
        p_value = case["P"]
        E_adim = E/140e9

        p_adim = (p_value * K_scale) / 140e9
        
        # 2. Théorie d'Euler-Bernoulli
        v_max_theory = (p_value * L**3) / (3 * E * I)
        
        E = torch.full_like(x_test, E_adim)
        p = torch.full_like(x_test, p_adim)
        nu_t = torch.full_like(x_test, nu)
        
        u, v = model(x_test, y_test, E, nu_t, p)
        
        u_pred = u / K_scale
        v_pred = v / K_scale

        
        v_max_pinn = abs(v_pred[0, 0].item()) # On prend la valeur absolue de la flèche mise à l'échelle
        
        # 5. Calcul de l'erreur
        error_relative = abs(v_max_pinn - v_max_theory) / v_max_theory
        
        print(f"\n--- Validation Physique A2S2 V0_298 ---")
        print(f"Flèche Théorique (Euler-Bernoulli) : {v_max_theory:.2e}")
        print(f"Flèche Prédite (PINN)             : {v_max_pinn:.2e}")
        print(f"Erreur Relative                   : {error_relative*100:.2f}%")
    
        # Seuil de tolérance : 1% pour une V0 (PINN 2D vs Théorie 1D)
        assert error_relative < 0.01, f"Erreur trop élevée : {error_relative*100:.2f}%"
    
