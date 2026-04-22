import torch
import torch.nn as nn
import numpy as np

class FourierEmbedding(nn.Module):
    """
    Mapping des coordonnées spatiales vers un espace de haute dimension (Fourier Features).
    Aide le réseau à capturer les hautes fréquences et les gradients abrupts (ex: bords d'attaque).
    """
    def __init__(self, in_features, n_freq=10):
        super().__init__()
        # On définit les fréquences (2^0, 2^1, ...)
        freqs = 2**torch.linspace(0, n_freq-1, n_freq)
        self.register_buffer('freqs', freqs)
        self.out_features = in_features * n_freq * 2
        

    def forward(self, x):
        # x shape: [N, in_features]
        # x_proj shape: [N, in_features, n_freq]
        x_proj = x.unsqueeze(-1) * self.freqs
        # Retourne concaténation de [sin(x*f), cos(x*f)]
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1).flatten(1)

class NeuralNet(nn.Module):
    """
    MLP flexible pour le solveur PINN A2S2.
    """
    def __init__(self, normalizer, input_dim=2, hidden_dim=32, num_layers=5, use_fourier=True):
        super(NeuralNet, self).__init__()

        self.normalizer = normalizer
        
        self.use_fourier = use_fourier

        def make_block(output_dim):

            if use_fourier:
                self.embedding = FourierEmbedding(input_dim, n_freq=4)
                current_dim = self.embedding.out_features
            else:
                current_dim = input_dim

            # Construction des couches cachées
            layers = []
            for i in range(num_layers):
                layers.append(nn.Linear(current_dim, hidden_dim))
                layers.append(nn.Tanh()) 
                current_dim = hidden_dim
            
            # Couche de sortie
            layers.append(nn.Linear(current_dim, output_dim))
            
            return nn.Sequential(*layers)

        self.net_u = make_block(1)
        self.net_v = make_block(1)


    def forward(self, x, y):
        """
        Calcul du forward pass.
        Note : x et y doivent avoir requires_grad=True pour le calcul des résidus PDE.
        """
        inputs = torch.cat([x, y], dim=1)
        inputs_norm = self.normalizer.encode(inputs)
        
        if self.use_fourier:
            inputs = self.embedding(inputs_norm)
        else : 
            inputs = inputs_norm

        u_raw = self.net_u(inputs)
        v_raw = self.net_v(inputs)

        # Masque sur les deplacement (hard_constraints)
        mask_u = 1.0
        mask_v = 1.0 
        u = mask_u * u_raw
        v = mask_v * v_raw

        return u, v
