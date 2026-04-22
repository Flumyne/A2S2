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
