import torch
import torch.optim as optim
from layers import NeuralNet
from pde_residuals import linear_elasticity_2d_residual, compute_stresses, get_gradients
from data_gen import Geometry
from utils import visualize_loss, Normalizer


class PINNSolver:
    """
    Gestionnaire d'entraînement ADAW (Phase 1) pour le solveur A2S2.
    """
    def __init__(self, model, n_points_col, n_points_bc, p, L, H, device, epochs, lr=1e-3, ):
        self.model = model
        self.optimizer_adamw = optim.AdamW(self.model.parameters(), lr=lr)
        self.optimizer_lbfgs = optim.LBFGS(self.model.parameters(), lr=1.0, max_iter = 5000, history_size=50, line_search_fn='strong_wolfe')
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_adamw, T_max=epochs, eta_min=1e-5)
        # Coefficients de pondération initiaux
        self.lambda_pde = 100.0
        self.lambda_bc_left = 100.0   
        self.lambda_bc_right = 100.0  
        self.lambda_bc_free = 1000.0  

        # Nombre de points à générer 
        self.n_points_col = n_points_col
        self.n_points_bc = n_points_bc
        self.p = p
        self.device = device
        self.geo = Geometry(L, H, device)

    def train_step_adamw(self, E, nu):
        """
        Effectue une itération d'entraînement.
        """
        self.optimizer_adamw.zero_grad()

        # 1. Génération des points 

        x_col, y_col = self.geo.generate_collocation_points(self.n_points_col)
        x_bc_top, y_bc_top = self.geo.generate_border_top(self.n_points_bc)
        x_bc_bot, y_bc_bot = self.geo.generate_border_bot(self.n_points_bc)
        x_bc_left, y_bc_left = self.geo.generate_border_left(self.n_points_bc)
        x_bc_right, y_bc_right = self.geo.generate_border_right(self.n_points_bc)

        # 2. Perte Physique (Résidus PDE) sur les points de collocation
        u_col_pred = self.model(x_col, y_col)
        res_x, res_y = linear_elasticity_2d_residual(u_col_pred[:, 0:1], u_col_pred[:, 1:2], x_col, y_col, E, nu)
        loss_pde = torch.mean(res_x**2 + res_y**2)

        # 3. Perte BC (Conditions aux limites)
        # 3.1 Left Dirichlet = 0
        u_bc_left_pred = self.model(x_bc_left, y_bc_left)
        dv_dx = get_gradients(u_bc_left_pred[:,1:2], x_bc_left)
        loss_bc_left = torch.mean((u_bc_left_pred[:, 0:1])**2 + (u_bc_left_pred[:, 1:2])**2 + (dv_dx)**2)

        # 3.2 Right Neumann sigma_xy = -p/H
        u_bc_right_pred = self.model(x_bc_right, y_bc_right)
        sigma_xx_right, _, sigma_xy_right_pred = compute_stresses(u_bc_right_pred[:, 0:1], u_bc_right_pred[:, 1:2], x_bc_right, y_bc_right, E, nu)
        # Profil parabolique : - (3*P)/(2*H) * (1 - (2*y/H)^2)
        y_norm = (2.0 * y_bc_right) / self.geo.H
        sigma_xy_right_real = - (3.0 * self.p) / (2.0 * self.geo.H) * (1.0 - y_norm**2)
        loss_bc_right = torch.mean((sigma_xy_right_pred - sigma_xy_right_real)**2 + (sigma_xx_right)**2)

        # 3.3 Top Neumann sigma_xy = 0 et sigma_yy = 0
        u_bc_top_pred = self.model(x_bc_top, y_bc_top)
        _, sigma_yy_top, sigma_xy_top = compute_stresses(u_bc_top_pred[:, 0:1], u_bc_top_pred[:, 1:2], x_bc_top, y_bc_top, E, nu)
        loss_bc_top = torch.mean((sigma_xy_top)**2 + (sigma_yy_top)**2 )

        # 3.4 Bot Neumann sigma_xy = 0 et sigma_yy = 0
        u_bc_bot_pred = self.model(x_bc_bot, y_bc_bot)
        _, sigma_yy_bot, sigma_xy_bot = compute_stresses(u_bc_bot_pred[:, 0:1], u_bc_bot_pred[:, 1:2], x_bc_bot, y_bc_bot, E, nu)
        loss_bc_bot = torch.mean((sigma_xy_bot)**2 + (sigma_yy_bot)**2 )

        loss_free = loss_bc_bot + loss_bc_top

        # 4. Pondération  
        total_loss = self.lambda_pde * loss_pde + self.lambda_bc_left * loss_bc_left + self.lambda_bc_right * loss_bc_right + self.lambda_bc_free * loss_free

        total_loss.backward()
        self.optimizer_adamw.step()

        return total_loss.item(), loss_pde.item(), loss_bc_left.item(), loss_bc_right.item(), loss_free.item()


    def train_lbfgs(self, E, nu, histories):
        """
        Gestionnaire d'entraînement LBFGS (Phase 2) pour le solveur A2S2.
        """

        loss_total_history, loss_pde_history, loss_bc_left_history, loss_bc_right_history, loss_bc_free_history = histories

        self.n_iter = 0

        def closure(): 
            self.optimizer_lbfgs.zero_grad()

            # 1. Génération des points 
            torch.manual_seed(42)
            x_col, y_col = self.geo.generate_collocation_points(self.n_points_col)
            x_bc_top, y_bc_top = self.geo.generate_border_top(self.n_points_bc)
            x_bc_bot, y_bc_bot = self.geo.generate_border_bot(self.n_points_bc)
            x_bc_left, y_bc_left = self.geo.generate_border_left(self.n_points_bc)
            x_bc_right, y_bc_right = self.geo.generate_border_right(self.n_points_bc)

            # 2. Perte Physique (Résidus PDE) sur les points de collocation
            u_col_pred = self.model(x_col, y_col)
            res_x, res_y = linear_elasticity_2d_residual(u_col_pred[:, 0:1], u_col_pred[:, 1:2], x_col, y_col, E, nu)
            loss_pde = torch.mean(res_x**2 + res_y**2)

            # 3. Perte BC (Conditions aux limites)
            # 3.1 Left Dirichlet = 0
            u_bc_left_pred = self.model(x_bc_left, y_bc_left)
            dv_dx = get_gradients(u_bc_left_pred[:,1:2], x_bc_left)
            loss_bc_left = torch.mean((u_bc_left_pred[:, 0:1])**2 + (u_bc_left_pred[:, 1:2])**2 + (dv_dx)**2)

            # 3.2 Right Neumann sigma_xy parabolique exact (Théorie des poutres)
            u_bc_right_pred = self.model(x_bc_right, y_bc_right)
            sigma_xx_right, _, sigma_xy_right_pred = compute_stresses(u_bc_right_pred[:, 0:1], u_bc_right_pred[:, 1:2], x_bc_right, y_bc_right, E, nu)
            
            # Profil parabolique : - (3*P)/(2*H) * (1 - (2*y/H)^2)
            y_norm = (2.0 * y_bc_right) / self.geo.H
            sigma_xy_right_real = - (3.0 * self.p) / (2.0 * self.geo.H) * (1.0 - y_norm**2)
            
            loss_bc_right = torch.mean((sigma_xy_right_pred - sigma_xy_right_real)**2 + (sigma_xx_right)**2)

            # 3.3 Top Neumann sigma_xy = 0 et sigma_yy = 0
            u_bc_top_pred = self.model(x_bc_top, y_bc_top)
            _, sigma_yy_top, sigma_xy_top = compute_stresses(u_bc_top_pred[:, 0:1], u_bc_top_pred[:, 1:2], x_bc_top, y_bc_top, E, nu)
            loss_bc_top = torch.mean((sigma_xy_top)**2 + (sigma_yy_top)**2 )

            # 3.4 Bot Neumann sigma_xy = 0 et sigma_yy = 0
            u_bc_bot_pred = self.model(x_bc_bot, y_bc_bot)
            _, sigma_yy_bot, sigma_xy_bot = compute_stresses(u_bc_bot_pred[:, 0:1], u_bc_bot_pred[:, 1:2], x_bc_bot, y_bc_bot, E, nu)
            loss_bc_bot = torch.mean((sigma_xy_bot)**2 + (sigma_yy_bot)**2 )

            loss_free = loss_bc_bot + loss_bc_top

            # 4. Pondération  
            total_loss = self.lambda_pde * loss_pde + self.lambda_bc_left * loss_bc_left + self.lambda_bc_right * loss_bc_right + self.lambda_bc_free * loss_free

            loss_total_history.append(total_loss.item())
            loss_pde_history.append(loss_pde.item())
            loss_bc_left_history.append(loss_bc_left.item())
            loss_bc_right_history.append(loss_bc_right.item())
            loss_bc_free_history.append(loss_free.item())

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

    model = NeuralNet(normalizer, input_dim=2, hidden_dim=128, output_dim=2, use_fourier=False).to(device)
    solver = PINNSolver(model, n_points_col=3000, n_points_bc=500, p=p, L=L, H = H, device=device, epochs=epochs)

    loss_total_history = []
    loss_pde_history = []
    loss_bc_left_history = []
    loss_bc_right_history = []
    loss_bc_free_history = []

    for i in range(epochs):
        total_loss, loss_pde, loss_bc_left, loss_bc_right, loss_bc_free = solver.train_step_adamw(E,nu)
        solver.scheduler.step()

        loss_total_history.append(total_loss)
        loss_pde_history.append(loss_pde)
        loss_bc_left_history.append(loss_bc_left)
        loss_bc_right_history.append(loss_bc_right)
        loss_bc_free_history.append(loss_bc_free)
        if i % 250 == 0:
            print(f"Epoch {i}: Loss = {total_loss:.2e} | Loss_PDE = {loss_pde:.2e} | Loss_BC_left = {loss_bc_left:.2e}| Loss_BC_right = {loss_bc_right:.2e}| Loss_BC_free = {loss_bc_free:.2e}")

    print("Démarrage du raffinement LBFGS...")
    histories = (loss_total_history, loss_pde_history, loss_bc_left_history, loss_bc_right_history, loss_bc_free_history)
    solver.train_lbfgs(E, nu, histories)


    torch.save(model.state_dict(), "A2S2_model_V0_07.pth")
    print("Modèle enregistré avec succès !")

    visualize_loss(loss_total_history, loss_pde_history, loss_bc_left_history, loss_bc_right_history, loss_bc_free_history)


