import torch
import matplotlib.pyplot as plt
import numpy

class Geometry: 
    """
    Classe pour la création du domaine géométrique et la génération de points de collocation
    """

    def __init__(self, L, H, device):
        self.L = L 
        self.H = H 
        self.x_range = [0, L]
        self.y_range = [-H/2, H/2]
        self.device = device

    def generate_collocation_points(self, n_points) : 
        """
        Génère les points de collocation à l'intérieur du domaine

        args : 
            n_points : nombre de points de collocation
            x_col, y_col : discrétisation des points aléatoirement dans le domaine
        """

        x_min = self.x_range[0]
        x_max = self.x_range[1]
        y_min = self.y_range[0]
        y_max = self.y_range[1]

        x_col = (x_max - x_min)*torch.rand(n_points, 1, device=self.device, requires_grad=True) + x_min
        y_col = (y_max - y_min)*torch.rand(n_points, 1, device=self.device, requires_grad=True) + y_min
        return x_col, y_col


    def generate_border_left(self, n_points):
        """
        Génère les points de boundary left

        args : 
            n_points : nombre de points 
            x_bc_left, y_bc_left : discrétisation des points sur la condition de gauche
        """

        x_bc_left = torch.ones(n_points, 1, device=self.device) * self.x_range[0]
        x_bc_left.requires_grad = True
        y_bc_left = torch.linspace(self.y_range[0], self.y_range[1], n_points, device=self.device, requires_grad=True).view(-1,1)
        return x_bc_left, y_bc_left

    def generate_border_top(self, n_points):
        """
        Génère les points de boundary Top

        args : 
            n_points : nombre de points 
            x_bc_top, y_bc_top : discrétisation des points sur la condition du haut
        """

        y_bc_top = torch.ones(n_points, 1, device=self.device) * self.y_range[1]
        y_bc_top.requires_grad = True
        x_bc_top = torch.linspace(self.x_range[0], self.x_range[1], n_points, device=self.device, requires_grad=True).view(-1,1) #.view permet de créer un tenseur (N,1) au lieu de (N,) renvoyer par linspace
        return x_bc_top, y_bc_top        

    def generate_border_bot(self, n_points):
        """
        Génère les points de boundary bot

        args : 
            n_points : nombre de points 
            x_bc_bot, y_bc_bot : discrétisation des points sur la condition du bas
        """

        y_bc_bot = torch.ones(n_points, 1, device=self.device) * self.y_range[0]
        y_bc_bot.requires_grad = True
        x_bc_bot = torch.linspace(self.x_range[0], self.x_range[1], n_points, device=self.device, requires_grad=True).view(-1,1) #.view permet de créer un tenseur (N,1) au lieu de (N,) renvoyer par linspace
        return x_bc_bot, y_bc_bot

    def generate_border_right(self, n_points):
        """
        Génère les points de boundary Right

        args : 
            n_points : nombre de points 
            x_bc_right, y_bc_right : discrétisation des points sur la condition de gauche
        """

        x_bc_right = torch.ones(n_points, 1, device=self.device) * self.x_range[1]
        x_bc_right.requires_grad = True
        y_bc_right = torch.linspace(self.y_range[0], self.y_range[1], n_points, device=self.device, requires_grad=True).view(-1,1) #.view permet de créer un tenseur (N,1) au lieu de (N,) renvoyer par linspace
        return x_bc_right, y_bc_right

    def visualize(self, n_points_col, n_points_bc): 
        """
        Permet la visualisation des points de discrétisation

        args : 
            n_points_col : nombre de points de collocation
            n_points_bc : nombre de points sur chaque bord
        """    

        x_col, y_col = self.generate_collocation_points(n_points_col)
        x_bc_top, y_bc_top = self.generate_border_top(n_points_bc)
        x_bc_bot, y_bc_bot = self.generate_border_bot(n_points_bc)
        x_bc_left, y_bc_left = self.generate_border_left(n_points_bc)
        x_bc_right, y_bc_right = self.generate_border_right(n_points_bc)

        fig, axes = plt.subplots(1, 1)
        axes.scatter(x_col.cpu().detach().numpy(), y_col.cpu().detach().numpy(), s=2, c='blue', label='Collocation')
        axes.scatter(x_bc_top.cpu().detach().numpy(), y_bc_top.cpu().detach().numpy(), s=2, c='red', label='TOP BC')
        axes.scatter(x_bc_bot.cpu().detach().numpy(), y_bc_bot.cpu().detach().numpy(), s=2, c='green', label='BOT BC')
        axes.scatter(x_bc_left.cpu().detach().numpy(), y_bc_left.cpu().detach().numpy(), s=2, c='yellow', label='LEFT BC')
        axes.scatter(x_bc_right.cpu().detach().numpy(), y_bc_right.cpu().detach().numpy(), s=2, c='purple', label='RIGHT BC')
        axes.set_xticks([])
        axes.set_yticks([])
        axes.set_aspect('equal')
        axes.set_xlim([self.x_range[0], self.x_range[1]])
        axes.set_ylim([self.y_range[0], self.y_range[1]])
        axes.legend(loc='upper center', bbox_to_anchor=(0.5,-0.15), ncol=2, frameon=False)
        axes.set_title(f"Points Position")
        plt.show()


if __name__ == "__main__" : 
    geo = Geometry(L=1.0, H=0.1)
    geo.visualize(n_points_col=2000, n_points_bc=200)