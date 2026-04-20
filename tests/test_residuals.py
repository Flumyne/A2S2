import torch
import sys
import os

# Ajout du chemin src pour l'import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.pde_residuals import linear_elasticity_2d_residual

def test_rigid_body_translation():
    """
    Un déplacement constant ne doit produire aucune contrainte 
    et donc un résidu nul.
    """
    x = torch.linspace(0, 1, 10, requires_grad=True).view(-1, 1)
    y = torch.linspace(0, 1, 10, requires_grad=True).view(-1, 1)
    
    # Déplacement constant u = 0.5, v = 0.2
    u = torch.full_like(x, 0.5).requires_grad_(True)
    v = torch.full_like(y, 0.2).requires_grad_(True)
    
    res_x, res_y = linear_elasticity_2d_residual(u, v, x, y)
    
    # Le résidu doit être très proche de 0
    assert torch.allclose(res_x, torch.zeros_like(res_x), atol=1e-7)
    assert torch.allclose(res_y, torch.zeros_like(res_y), atol=1e-7)

def test_constant_strain():
    """
    Un champ de déplacement linéaire produit une déformation constante.
    Ex: u = 0.01 * x  => eps_xx = 0.01.
    Les dérivées secondes (résidus) doivent être nulles.
    """
    x = torch.linspace(0, 1, 10, requires_grad=True).view(-1, 1)
    y = torch.linspace(0, 1, 10, requires_grad=True).view(-1, 1)
    
    u = (0.01 * x).requires_grad_(True)
    v = (0.005 * y).requires_grad_(True)
    
    res_x, res_y = linear_elasticity_2d_residual(u, v, x, y)
    
    assert torch.allclose(res_x, torch.zeros_like(res_x), atol=1e-7)
    assert torch.allclose(res_y, torch.zeros_like(res_y), atol=1e-7)
