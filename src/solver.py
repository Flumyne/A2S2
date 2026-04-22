import torch
import torch.optim as optim
from layers import NeuralNet
from pde_residuals import linear_elasticity_2d_residual, compute_stresses, linear_elasticity_mixed_residual, compute_strain_energy
from data_gen import Geometry
from utils import visualize_loss, Normalizer


class PINNSolver:
    """
    Gestionnaire d'entraînement ADAW (Phase 1) pour le solveur A2S2.
    """
    def __init__(self, model, n_points_col, n_points_bc, p, L, H, device, epochs, lr=1e-3, ):
        self.model = model
        self.optimizer_adamw = optim.AdamW(self.model.parameters(), lr=lr)
        self.optimizer_lbfgs = optim.LBFGS(self.model.parameters(), lr=1e-1, max_iter = 5000, history_size=50, line_search_fn='strong_wolfe')
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_adamw, T_max=epochs, eta_min=1e-5)

        # Nombre de points à générer 
        self.n_points_col = n_points_col
        self.n_points_bc = n_points_bc
        self.p = p
        self.device = device
        self.geo = Geometry(L, H, device)

    def compute_loss(self, E, nu, x_col, y_col, x_bc_right, y_bc_right, x_bc_left, y_bc_left):

        # 1. Energie interne (sur tout le domaine)
        u_col, v_col = self.model(x_col, y_col)
        W_int = compute_strain_energy(u_col, v_col, x_col, y_col, E, nu)
        U = torch.mean(W_int)

        # 2. Travail Externe (Sur les bords)
        u_r, v_r = self.model(x_bc_right, y_bc_right)
        y_norm = (2.0 * y_bc_right)/self.geo.H
        sxy_target = - (3.0 * self.p) / (2.0*self.geo.H)* (1.0 - y_norm**2)

        # Travail = Force * Déplacement (force selon y donc v_r)
        # On divise par L pour normaliser par rapport au volume (puisque Volume = Surface * L)
        W_ext = torch.mean(sxy_target * v_r) / self.geo.L

        loss_energy = U - W_ext

        # 3. Dirichlet à gauche
        u_l, v_l = self.model(x_bc_left, y_bc_left)
        loss_bc_left = torch.mean(u_l**2 + v_l**2)

        return loss_energy, loss_bc_left, U, W_ext

    def train_step_adamw(self, E, nu):
        """
        Effectue une itération d'entraînement.
        """
        self.optimizer_adamw.zero_grad()

        # 1. Génération des points 

        x_col, y_col = self.geo.generate_collocation_points(self.n_points_col)
        x_bc_left, y_bc_left = self.geo.generate_border_left(self.n_points_bc)
        x_bc_right, y_bc_right = self.geo.generate_border_right(self.n_points_bc)

        # 2. Calcul de la perte
        loss_energy, loss_bc_left, U, W_ext = self.compute_loss(E, nu, x_col, y_col, x_bc_right, y_bc_right, x_bc_left, y_bc_left)

        # 3. Pondération  
        total_loss = loss_energy + 1e3 * loss_bc_left

        total_loss.backward()
        self.optimizer_adamw.step()

        return total_loss.item(), loss_energy.item(), loss_bc_left.item()


    def train_lbfgs(self, E, nu, histories):
        """
        Gestionnaire d'entraînement LBFGS (Phase 2) pour le solveur A2S2.
        """

        loss_total_history, loss_energy_history, loss_bc_left_history = histories

        self.n_iter = 0

        def closure(): 
            self.optimizer_lbfgs.zero_grad()

            # 1. Génération des points 
            torch.manual_seed(42)
            x_col, y_col = self.geo.generate_collocation_points(self.n_points_col)
            x_bc_left, y_bc_left = self.geo.generate_border_left(self.n_points_bc)
            x_bc_right, y_bc_right = self.geo.generate_border_right(self.n_points_bc)

            # 2. Calcul de la perte
            loss_energy, loss_bc_left, U, W_ext = self.compute_loss(E, nu, x_col, y_col, x_bc_right, y_bc_right, x_bc_left, y_bc_left)

            # 3. Pondération  
            total_loss = loss_energy + 1e3 * loss_bc_left

            loss_total_history.append(total_loss.item())
            loss_bc_left_history.append(loss_bc_left.item())
            loss_energy_history.append(loss_energy.item())

            self.n_iter += 1
            if self.n_iter % 100 == 0:
                print(f"LBFGS Iter {self.n_iter}: Loss = {total_loss.item():.2e}")

            total_loss.backward()

            return total_loss

        self.optimizer_lbfgs.step(closure)

if __name__ == "__main__":
    # Test 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entraînement sur : {device}")

    E_ref = 70e9
    nu = 0.33
    L_ref = 1.0 
    H = 0.1
    p = 1000

    L = 1.0
    H = H / L_ref
    E = 1.0
    
    # LOAD SCALING : On multiplie la force par 100 000 pour éviter l'underflow Float32
    K_scale = 1e5
    p = (p * K_scale) / E_ref

    epochs = 10001

    geo = Geometry(L, H, device)
    x_sample, y_sample = geo.generate_collocation_points(5000)
    X_sample = torch.cat([x_sample,y_sample], dim=1)

    normalizer = Normalizer(X_sample, device=device)
    print(f"Stats Normalisation - Mean: {normalizer.mean.cpu().numpy()}, Std: {normalizer.std.cpu().numpy()}")

    model = NeuralNet(normalizer, input_dim=2, hidden_dim=50, use_fourier=False).to(device)
    solver = PINNSolver(model, n_points_col=3000, n_points_bc=500, p=p, L=L, H = H, device=device, epochs=epochs)

    loss_total_history = []
    loss_energy_history = []
    loss_bc_left_history = []


    for i in range(epochs):
        total_loss, loss_energy, loss_bc_left = solver.train_step_adamw(E,nu)
        solver.scheduler.step()

        loss_total_history.append(total_loss)
        loss_energy_history.append(loss_energy)
        loss_bc_left_history.append(loss_bc_left)


        if i % 250 == 0:
            print(f"Epoch {i}: Loss = {total_loss:.2e} ")

    print("Démarrage du raffinement LBFGS...")
    histories = (loss_total_history, loss_energy_history, loss_bc_left_history)
    solver.train_lbfgs(E, nu, histories)


    torch.save(model.state_dict(), "A2S2_model_V0_2.pth")
    print("Modèle enregistré avec succès !")

    visualize_loss(loss_total_history, loss_energy_history, loss_bc_left_history)


