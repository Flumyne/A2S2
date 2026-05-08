import torch
import torch.optim as optim
from layers import NeuralNet
from pde_residuals import compute_strain_energy
from data_gen import Geometry
from utils import visualize_loss, Normalizer
import numpy as np
import math


class PINNSolver:
    """
    Gestionnaire d'entraînement ADAW (Phase 1) pour le solveur A2S2.
    """
    def __init__(self, model, n_points_col, n_points_bc, L, H, device, epochs, lr=1e-3, ):
        self.model = model
        self.optimizer_adamw = optim.AdamW(self.model.parameters(), lr=lr)
        self.optimizer_adamw_transi = optim.AdamW(self.model.parameters(), lr=1e-5)
        self.optimizer_lbfgs = optim.LBFGS(self.model.parameters(), lr=1e-2, max_iter = 5000, history_size=50, line_search_fn='strong_wolfe')
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_adamw, T_max=epochs, eta_min=1e-5)

        # Nombre de points à générer 
        self.n_points_col = n_points_col
        self.n_points_bc = n_points_bc
        self.device = device
        self.geo = Geometry(L, H, device)

    def compute_loss(self, E_list, nu_list, p_list, x_col, y_col, x_bc_right, y_bc_right, x_bc_left, y_bc_left):
        
        def prepare_params(target_value, val_list):
            # val_list est un tenseur 1D de taille N (nombre de cas)
            # target_value a une taille totale = N * points_par_cas
            N = len(val_list)
            points_par_cas = target_value.shape[0] // N
            # On étend val_list pour qu'il matche parfaitement chaque bloc de points
            val_tensor = val_list.view(N, 1, 1).expand(N, points_par_cas, 1)
            return val_tensor.reshape(-1, 1)
            
        E_col = prepare_params(x_col, E_list)
        nu_col = prepare_params(x_col, nu_list)
        p_col = prepare_params(x_col, p_list)

        E_r = prepare_params(x_bc_right, E_list)
        nu_r = prepare_params(x_bc_right, nu_list)
        p_r = prepare_params(x_bc_right, p_list)

        E_l = prepare_params(x_bc_left, E_list)
        nu_l = prepare_params(x_bc_left, nu_list)
        p_l = prepare_params(x_bc_left, p_list)


        scale_energy_col = p_col**2 / E_col
        scale_energy_r = p_r**2 / E_r
        scale_disp_l = p_l**2 / E_l**2

        # 1. Energie interne (sur tout le domaine)
        u_col, v_col = self.model(x_col, y_col, E_col, nu_col, p_col)
        W_int = compute_strain_energy(u_col, v_col, x_col, y_col, E_col, nu_col)
        U = torch.mean(W_int/scale_energy_col )

        # 2. Travail Externe (Sur les bords)
        u_r, v_r = self.model(x_bc_right, y_bc_right, E_r, nu_r, p_r)
        y_norm = (2.0 * y_bc_right)/self.geo.H
        sxy_target = - (3.0 * p_r) / (2.0*self.geo.H)* (1.0 - y_norm**2)

        W_ext = torch.mean((sxy_target * v_r) / (scale_energy_r)) / (self.geo.L)

        loss_energy = U - W_ext

        # 3. Dirichlet à gauche
        u_l, v_l = self.model(x_bc_left, y_bc_left, E_l, nu_l, p_l)
        loss_bc_left = torch.mean((u_l**2 + v_l**2)/ scale_disp_l)

        return loss_energy, loss_bc_left, U, W_ext

    def generate_batch(self, batch_per_case, n_random_case): 

        all_x_col, all_y_col = [], []
        all_x_bc_l, all_y_bc_l, all_x_bc_r, all_y_bc_r = [], [], [], []
        
        K_scale = 1e5

        if n_random_case > 0: 
            # Mode AdamW : Tirage aléatoire de N cas 
            log_E = torch.rand(n_random_case) * (math.log(140e9) - math.log(35e9)) + math.log(35e9)
            log_p = torch.rand(n_random_case) * (math.log(5000) - math.log(500)) + math.log(500)

            E_ref_list = torch.exp(log_E).to(self.device)
            nu_list = (0.45 - 0.1) * torch.rand(n_random_case).to(self.device) + 0.1
            p_list =  torch.exp(log_p).to(self.device)

            E_list = E_ref_list / 140e9
            p_list = (p_list * K_scale) / 140e9

            for i in range(n_random_case):
                x_sample, y_sample = self.geo.generate_collocation_points(batch_per_case)
                x_bc_left, y_bc_left = self.geo.generate_border_left(int(batch_per_case/10))
                x_bc_right, y_bc_right = self.geo.generate_border_right(int(batch_per_case/10))

                all_x_col.append(x_sample)
                all_y_col.append(y_sample)
                all_x_bc_l.append(x_bc_left)
                all_y_bc_l.append(y_bc_left)
                all_x_bc_r.append(x_bc_right)
                all_y_bc_r.append(y_bc_right)

        else : 
            # Mode LBFGS et Transitionnel: Grille Fixe 3D 
            E_ref_grid = np.linspace(35e9, 140e9, 15)
            p_grid_raw = np.linspace(500, 5000, 15)
            nu_grid = np.linspace(0.1, 0.45, 3)

            E_grid = E_ref_grid / 140e9
            p_grid = (p_grid_raw * K_scale) / 140e9
            
            E_list, nu_list, p_list = [], [], []

            for E in E_grid:
                for nu in nu_grid: 
                    for p in p_grid:
                        x_sample, y_sample = self.geo.generate_collocation_points(batch_per_case)
                        x_bc_left, y_bc_left = self.geo.generate_border_left(int(batch_per_case/10))
                        x_bc_right, y_bc_right = self.geo.generate_border_right(int(batch_per_case/10))

                        all_x_col.append(x_sample)
                        all_y_col.append(y_sample)
                        all_x_bc_l.append(x_bc_left)
                        all_y_bc_l.append(y_bc_left)
                        all_x_bc_r.append(x_bc_right)
                        all_y_bc_r.append(y_bc_right)
                        
                        E_list.append(E)
                        nu_list.append(nu)
                        p_list.append(p)
            
            E_list = torch.tensor(E_list, dtype=torch.float32, device=self.device)
            nu_list = torch.tensor(nu_list, dtype=torch.float32, device=self.device)
            p_list = torch.tensor(p_list, dtype=torch.float32, device=self.device)

        fix_x_col = torch.cat(all_x_col).detach().requires_grad_(True)
        fix_y_col = torch.cat(all_y_col).detach().requires_grad_(True)
        fix_x_bc_l = torch.cat(all_x_bc_l).detach().requires_grad_(True)
        fix_y_bc_l = torch.cat(all_y_bc_l).detach().requires_grad_(True)
        fix_x_bc_r = torch.cat(all_x_bc_r).detach().requires_grad_(True)
        fix_y_bc_r = torch.cat(all_y_bc_r).detach().requires_grad_(True)

        return fix_x_col, fix_y_col, fix_x_bc_l, fix_y_bc_l, fix_x_bc_r, fix_y_bc_r, E_list, nu_list, p_list



    def train_step_adamw(self):
        """
        Effectue une itération d'entraînement.
        """
        self.optimizer_adamw.zero_grad()

        # 1. Génération des points 

        fix_x_col, fix_y_col, fix_x_bc_l, fix_y_bc_l, fix_x_bc_r, fix_y_bc_r, fix_E, fix_nu, fix_p = self.generate_batch(800, 32)

        # 2. Calcul de la perte
        loss_energy, loss_bc_left, U, W_ext = self.compute_loss(fix_E, fix_nu, fix_p, fix_x_col, fix_y_col, fix_x_bc_r, fix_y_bc_r, fix_x_bc_l, fix_y_bc_l)

        # 3. Pondération  
        total_loss = loss_energy + 1e2 * loss_bc_left

        total_loss.backward()
        self.optimizer_adamw.step()

        return total_loss.item(), loss_energy.item(), loss_bc_left.item(), U.item(), W_ext.item()

    def transition_adamw_to_lbfgs(self):
        """
        Phase de transition entre AdamW et LBFGS.
        """

        self.optimizer_adamw_transi.zero_grad()

        # 1. Génération des points
        torch.manual_seed(42)
        fix_x_col, fix_y_col, fix_x_bc_l, fix_y_bc_l, fix_x_bc_r, fix_y_bc_r, fix_E, fix_nu, fix_p = self.generate_batch(300, 0)

        # 2. Calcul de la perte
        loss_energy, loss_bc_left, U, W_ext = self.compute_loss(fix_E, fix_nu, fix_p, fix_x_col, fix_y_col, fix_x_bc_r, fix_y_bc_r, fix_x_bc_l, fix_y_bc_l)

        # 3. Pondération  
        total_loss = loss_energy + 1e2 * loss_bc_left

        total_loss.backward()
        self.optimizer_adamw_transi.step()
        

        return total_loss.item(), loss_energy.item(), loss_bc_left.item(), U.item(), W_ext.item()


    def train_lbfgs(self, histories):
        """
        Gestionnaire d'entraînement LBFGS (Phase 2) pour le solveur A2S2.
        """

        loss_total_history, loss_energy_history, loss_bc_left_history, loss_U_history, loss_W_ext_history = histories



        self.n_iter = 0

        def closure(): 
            self.optimizer_lbfgs.zero_grad()

            # 1. Génération des points 
            torch.manual_seed(42)
            fix_x_col, fix_y_col, fix_x_bc_l, fix_y_bc_l, fix_x_bc_r, fix_y_bc_r, fix_E, fix_nu, fix_p = self.generate_batch(300, 0)

            # 2. Calcul de la perte
            loss_energy, loss_bc_left, U, W_ext = self.compute_loss(fix_E, fix_nu, fix_p, fix_x_col, fix_y_col, fix_x_bc_r, fix_y_bc_r, fix_x_bc_l, fix_y_bc_l)

            # 3. Pondération  
            total_loss = loss_energy + 1e2 * loss_bc_left

            loss_total_history.append(total_loss.item())
            loss_bc_left_history.append(loss_bc_left.item())
            loss_energy_history.append(loss_energy.item())
            loss_U_history.append(U.item())
            loss_W_ext_history.append(W_ext.item())

            self.n_iter += 1
            if self.n_iter % 100 == 0:
                print(f"LBFGS Iter {self.n_iter}: Loss = {total_loss.item():.2e}")

            total_loss.backward()

            return total_loss

        self.optimizer_lbfgs.step(closure)

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    # Test 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entraînement sur : {device}")

    L_ref = 1.0 
    H = 0.1
    L = 1.0
    H = H / L_ref

    E_ref_sample = (140e9 - 35e9)*torch.rand(5000,1).to(device) + 35e9
    nu_sample = (0.45 - 0.1)*torch.rand(5000,1).to(device) + 0.1
    p_sample = (5000 - 500)*torch.rand(5000,1).to(device) + 500
    E_sample = E_ref_sample/140e9
    # LOAD SCALING : On multiplie la force par 100 000 pour éviter l'underflow Float32
    K_scale = 1e5
    p_sample = (p_sample * K_scale) / 140e9

    epochs = 20001

    geo = Geometry(L, H, device)
    x_sample, y_sample = geo.generate_collocation_points(5000)
    X_sample_spatial = torch.cat([x_sample,y_sample], dim=1)
    X_sample_param = torch.cat([E_sample, nu_sample, p_sample], dim=1)

    normalizer_spatial = Normalizer(X_sample_spatial, device=device)
    normalizer_param = Normalizer(X_sample_param, device=device)
    print(f"Stats Normalisation Spatial - Mean: {normalizer_spatial.mean.cpu().numpy()}, Std: {normalizer_spatial.std.cpu().numpy()}")
    print(f"Stats Normalisation Params - Mean: {normalizer_param.mean.cpu().numpy()}, Std: {normalizer_param.std.cpu().numpy()}")

    model = NeuralNet(normalizer_spatial, normalizer_param, hidden_dim=100, use_fourier=True).to(device)
    solver = PINNSolver(model, n_points_col=3000, n_points_bc=500, L=L, H = H, device=device, epochs=epochs)

    loss_total_history = []
    loss_energy_history = []
    loss_bc_left_history = []
    loss_U_history = []
    loss_W_ext_history = []

    print("Démarrage de la phase 1 AdamW...")

    for i in range(epochs):
        
        total_loss, loss_energy, loss_bc_left, U, W_ext = solver.train_step_adamw()
        solver.scheduler.step()

        loss_total_history.append(total_loss)
        loss_energy_history.append(loss_energy)
        loss_bc_left_history.append(loss_bc_left)
        loss_U_history.append(U)
        loss_W_ext_history.append(W_ext)


        if i % 250 == 0:
            print(f"Epoch {i}: Loss = {total_loss:.2e} ")

    '''
    print("Démarrage de la phase de transition AdamW...")

    for i in range(51):
        
        total_loss, loss_energy, loss_bc_left, U, W_ext = solver.transition_adamw_to_lbfgs()

        loss_total_history.append(total_loss)
        loss_energy_history.append(loss_energy)
        loss_bc_left_history.append(loss_bc_left)
        loss_U_history.append(U)
        loss_W_ext_history.append(W_ext)


        if i % 25 == 0:
            print(f"Epoch {i}: Loss = {total_loss:.2e} ")        
    '''
    print("Démarrage du raffinement LBFGS...")
    torch.cuda.empty_cache()
    histories = (loss_total_history, loss_energy_history, loss_bc_left_history, loss_U_history, loss_W_ext_history)
    solver.train_lbfgs(histories)

    torch.save(model.state_dict(), "A2S2_model_V0_298.pth")
    print("Modèle enregistré avec succès !")

    visualize_loss(loss_total_history, loss_energy_history, loss_bc_left_history, loss_U_history, loss_W_ext_history)


