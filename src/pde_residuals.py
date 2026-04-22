import torch

def get_gradients(u, x):
    """Calcule le gradient de u par rapport à x en utilisant Autograd."""
    if not u.requires_grad or u.grad_fn is None :
        return torch.zeros_like(x)

    grads = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),create_graph=True,retain_graph=True, allow_unused=True)[0]
    if grads is None:
        return torch.zeros_like(x)
    return grads    

def compute_stresses(u, v, x, y, E, nu): 
    """
    Calcule des contraintes
    Hypothèse : Contraintes Planes (Plane Stress).

    Args:
        u, v : Déplacements prédits (sorties du NN)
        x, y : Coordonnées d'entrée (nécessitent requires_grad=True)
        E : Module de Young
        nu : Coefficient de Poisson
    """
    # 1. Dérivées des déplacements (Gradients)
    u_x = get_gradients(u, x)
    u_y = get_gradients(u, y)
    v_x = get_gradients(v, x)
    v_y = get_gradients(v, y)
    
    # 2. Tenseur des déformations (Strains) - Infinitésimal
    eps_xx = u_x
    eps_yy = v_y
    eps_xy = 0.5 * (u_y + v_x)
    
    # 3. Tenseur des contraintes (Stresses) - Loi de Hooke (Plane Stress)
    # sigma_xx = (E / (1 - nu^2)) * (eps_xx + nu * eps_yy)
    # sigma_yy = (E / (1 - nu^2)) * (eps_yy + nu * eps_xx)
    # sigma_xy = (E / (1 + nu)) * eps_xy
    
    prefactor = E / (1 - nu**2)
    
    sig_xx = prefactor * (eps_xx + nu * eps_yy)
    sig_yy = prefactor * (eps_yy + nu * eps_xx)
    sig_xy = (E / (1 + nu)) * eps_xy 
    return sig_xx, sig_yy, sig_xy


def linear_elasticity_2d_residual(u, v, x, y, E=1.0, nu=0.3):
    """
    Calcule les résidus des équations de Navier-Cauchy (Élasticité Linéaire 2D).
   
    Args:
        u, v : Déplacements prédits (sorties du NN)
        x, y : Coordonnées d'entrée (nécessitent requires_grad=True)
        E : Module de Young
        nu : Coefficient de Poisson
    """

    # 1. Calcul des contraintes
    sig_xx, sig_yy, sig_xy = compute_stresses(u, v, x, y, E, nu)
    
    # 2. Équations d'Équilibre (Résidus PDE)
    # d(sig_xx)/dx + d(sig_xy)/dy = 0
    # d(sig_xy)/dx + d(sig_yy)/dy = 0
    
    sig_xx_x = get_gradients(sig_xx, x)
    sig_xy_y = get_gradients(sig_xy, y)
    
    sig_xy_x = get_gradients(sig_xy, x)
    sig_yy_y = get_gradients(sig_yy, y)
    
    res_x = sig_xx_x + sig_xy_y
    res_y = sig_xy_x + sig_yy_y
    
    return res_x, res_y


def linear_elasticity_mixed_residual(u, v, sxx, syy, sxy, x, y, E=1.0, nu=0.3):
    # 1. Résidus d'équilibre 
    res_eq_x = get_gradients(sxx, x) + get_gradients(sxy, y)
    res_eq_y = get_gradients(sxy, x) + get_gradients(syy, y)

    # 2. Résidus Constitutifs (Cohérence avec Hooke)
    sxx_h, syy_h, sxy_h = compute_stresses(u, v, x, y, E, nu)
    
    res_c_xx = sxx - sxx_h
    res_c_yy = syy - syy_h
    res_c_xy = sxy - sxy_h

    return res_eq_x, res_eq_y, res_c_xx, res_c_yy, res_c_xy


def compute_strain_energy(u, v, x, y, E, nu):

    # 1. Calcul des gradients
    u_x = get_gradients(u, x)
    u_y = get_gradients(u, y)
    v_x = get_gradients(v, x)
    v_y = get_gradients(v, y)

    # 2. Calcul des déformations 

    esp_xx = u_x 
    esp_yy = v_y 
    gamma_xy = u_y + v_x

    # 3. Calcul de la densité d'énergie de déformation (W_int)
    prefactor = E / (2.0* (1.0 - nu**2))
    W_int = prefactor* (esp_xx**2 + esp_yy**2 + 2*nu*esp_xx*esp_yy + ((1-nu)/2)*gamma_xy**2)

    return W_int



if __name__ == "__main__":
    # Test rapide de la forme des tenseurs
    x = torch.linspace(0, 1, 10, requires_grad=True).view(-1, 1)
    y = torch.linspace(0, 1, 10, requires_grad=True).view(-1, 1)
    
    # Simuler des sorties de réseau de neurones
    u = x**2 + y**2
    v = x**2 - y**2
    
    rx, ry = linear_elasticity_2d_residual(u, v, x, y)
    print(f"Residual X shape: {rx.shape}")
    print(f"Residual Y shape: {ry.shape}")
