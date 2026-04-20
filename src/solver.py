import torch
import torch.optim as optim
from layers import NeuralNet
from pde_residuals import linear_elasticity_2d_residual, compute_stresses
from data_gen import Geometry
from utils import visualize_loss


class PINNSolver:
    """
    Gestionnaire d'entraînement pour le solveur A2S2.
    Implémente la pondération adaptative des pertes (Physics vs Boundary Conditions).
    """
    def __init__(self, model, n_points_col, n_points_bc, p, L, H, device, lr=1e-3):
        self.model = model
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        # Coefficients de pondération initiaux
        self.lambda_pde = 1.0
        self.lambda_bc = 2.0 
        # Nombre de points à générer 
        self.n_points_col = n_points_col
        self.n_points_bc = n_points_bc
        self.p = p
        self.device = device
        self.geo = Geometry(L, H, device)



    def train_step(self, E, nu):
        """
        Effectue une itération d'entraînement.
        """
        self.optimizer.zero_grad()

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
        loss_bc_left = torch.mean((u_bc_left_pred[:, 0:1])**2 + (u_bc_left_pred[:, 1:2])**2)

        # 3.2 Right Neumann sigma_xy = -p/H
        u_bc_right_pred = self.model(x_bc_right, y_bc_right)
        sigma_xx_right, _, sigma_xy_right_pred = compute_stresses(u_bc_right_pred[:, 0:1], u_bc_right_pred[:, 1:2], x_bc_right, y_bc_right, E, nu)
        sigma_xy_right_real = -self.p/self.geo.H
        loss_bc_right = torch.mean((sigma_xy_right_pred - sigma_xy_right_real)**2 + (sigma_xx_right)**2)

        # 3.3 Top Neumann sigma_xy = 0 et sigma_yy = 0
        u_bc_top_pred = self.model(x_bc_top, y_bc_top)
        _, sigma_yy_top, sigma_xy_top = compute_stresses(u_bc_top_pred[:, 0:1], u_bc_top_pred[:, 1:2], x_bc_top, y_bc_top, E, nu)
        loss_bc_top = torch.mean((sigma_xy_top)**2 + (sigma_yy_top)**2 )

        # 3.4 Bot Neumann sigma_xy = 0 et sigma_yy = 0
        u_bc_bot_pred = self.model(x_bc_bot, y_bc_bot)
        _, sigma_yy_bot, sigma_xy_bot = compute_stresses(u_bc_bot_pred[:, 0:1], u_bc_bot_pred[:, 1:2], x_bc_bot, y_bc_bot, E, nu)
        loss_bc_bot = torch.mean((sigma_xy_bot)**2 + (sigma_yy_bot)**2 )

        loss_bc = loss_bc_bot + loss_bc_left + loss_bc_right + loss_bc_top 

        # 4. Pondération  
        total_loss = self.lambda_pde * loss_pde + self.lambda_bc * loss_bc

        total_loss.backward()
        self.optimizer.step()

        return total_loss.item(), loss_pde.item(), loss_bc.item()

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
    p = p / E_ref

    model = NeuralNet(input_dim=2, hidden_dim=32, output_dim=2).to(device)
    solver = PINNSolver(model, n_points_col=3000, n_points_bc=300, p=p, L=L, H = H, device=device)

    loss_total_history = []
    loss_pde_history = []
    loss_bc_history = []

    epochs = 5000
    for i in range(epochs):
        total_loss, loss_pde, loss_bc = solver.train_step(E,nu)

        loss_total_history.append(total_loss)
        loss_pde_history.append(loss_pde)
        loss_bc_history.append(loss_bc)
        if i % 250 == 0:
            print(f"Epoch {i}: Loss = {total_loss:.2e} | Loss_PDE = {loss_pde:.2e} | Loss_BC = {loss_bc:.2e}")


    torch.save(model.state_dict(), "A2S2_model_V0.pth")
    print("Modèle enregistré avec succès !")

    visualize_loss(loss_total_history, loss_pde_history, loss_bc_history)


