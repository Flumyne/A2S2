import torch
import matplotlib.pyplot as plt 
from layers import NeuralNet
from pde_residuals import compute_stresses
from utils import Normalizer
from data_gen import Geometry


def run_inference(model_path, L, H, E, nu, device): 
    """
    Génère et affiche les champs de déplacement et de contrainte.
    """

    geo = Geometry(L, H, device)
    x_sample, y_sample = geo.generate_collocation_points(3000)
    X_sample = torch.cat([x_sample,y_sample], dim=1)

    normalizer = Normalizer(X_sample, device=device)

    # 1. Chargement du modèle
    model = NeuralNet(normalizer, input_dim=2, hidden_dim=128, output_dim=2, use_fourier=False).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 2. Création de la grille 2D
    x_vec = torch.linspace(0.0, L, 300, device=device)
    y_vec = torch.linspace(-H/2, H/2, 100, device=device)
    X, Y = torch.meshgrid(x_vec, y_vec, indexing='ij')
    
    x_flat = X.reshape(-1, 1).requires_grad_(True)
    y_flat = Y.reshape(-1, 1).requires_grad_(True)

    # 3. Calcul de u,v et des contraintes (en mode Tenseur)
    u_v = model(x_flat, y_flat)
    u_pred = u_v[:, 0:1]
    v_pred = u_v[:, 1:2]
    
    sig_xx, sig_yy, sig_xy = compute_stresses(u_pred, v_pred, x_flat, y_flat, E, nu)

    # 4. Calcul Von Mises (en Tenseur)
    # sigma_vm = sqrt(sig_xx^2 - sig_xx*sig_yy + sig_yy^2 + 3*sig_xy^2)
    vm_tensor = torch.sqrt(sig_xx**2 - sig_xx*sig_yy + sig_yy**2 + 3*sig_xy**2)

    # 5. Conversion vers Numpy pour Matplotlib
    x_plot = x_flat.detach().cpu().numpy()
    y_plot = y_flat.detach().cpu().numpy()
    K_scale = 1e5
    u_plot_val = u_pred.detach().cpu().numpy() / K_scale
    v_plot_val = v_pred.detach().cpu().numpy() / K_scale
    vm_plot_val = vm_tensor.detach().cpu().numpy() / K_scale

    # --- Plotting ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    # 1. Déplacement Horizontal u
    u_map = axes[0].scatter(x_plot, y_plot, c=u_plot_val, cmap="jet", s=10)
    axes[0].set_title("Déplacement Horizontal (u)")
    fig.colorbar(u_map, ax=axes[0])

    # 2. Déplacement Vertical v (Flèche)
    v_map = axes[1].scatter(x_plot, y_plot, c=v_plot_val, cmap="jet", s=10)
    axes[1].set_title("Déplacement Vertical (v)")
    fig.colorbar(v_map, ax=axes[1])

    # 3. Contrainte de Von Mises
    vm_map = axes[2].scatter(x_plot, y_plot, c=vm_plot_val, cmap="magma", s=10)
    axes[2].set_title("Contrainte de Von Mises (Stress)")
    fig.colorbar(vm_map, ax=axes[2])

    for ax in axes:
        ax.set_aspect('equal')
        ax.set_xlim([0, L])
        ax.set_ylim([-H/2, H/2])

    plt.tight_layout()
    output_name = "Field_Structure_A2S2_V0_07.png"
    plt.savefig(output_name, dpi=200)
    print(f"Visualisation sauvegardée sous : {output_name}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paramètres de normalisation (identiques au solver)
    E_ref = 70e9
    L_ref = 1.0 
    H_ref = 0.1
    p_ref = 1000

    # Valeurs adimensionnelles
    L = 1.0
    H = H_ref / L_ref
    E = 1.0
    nu = 0.33
    
    run_inference('A2S2_model_V0_07.pth', L, H, E, nu, device)
