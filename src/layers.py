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
    def __init__(self, normalizer_spatial, normalizer_param, hidden_dim=32, num_layers=5, use_fourier=True):
        super(NeuralNet, self).__init__()

        self.normalizer_s = normalizer_spatial
        self.normalizer_p = normalizer_param
        
        self.use_fourier = use_fourier

        input_dim_s = 2
        input_dim_p = 3

        if use_fourier:
            self.embedding = FourierEmbedding(input_dim_s, n_freq=2)
            current_dim = self.embedding.out_features
        else:
            current_dim = input_dim_s

        self.spatial_net = self.make_block(current_dim, hidden_dim, num_layers)
        self.params_net = self.make_block(input_dim_p, 16, 2)

        self.combined_net = nn.Sequential(
            nn.Linear(hidden_dim + 16, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2)
            )


    def make_block(self,input_dim, output_dim, num_layers):

        # Construction des couches cachées
        layers = []
        current_dim = input_dim
        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, output_dim))
            layers.append(nn.Tanh()) 
            current_dim = output_dim
        return nn.Sequential(*layers)

    def forward(self, x, y, E, nu, p):
        """
        Calcul du forward pass.
        Note : x et y doivent avoir requires_grad=True pour le calcul des résidus PDE.
        """
        inputs_s = torch.cat([x, y], dim=1)
        inputs_p = torch.cat([E, nu, p], dim=1)
        inputs_s_norm = self.normalizer_s.encode(inputs_s)
        inputs_p_norm = self.normalizer_p.encode(inputs_p)
        
        if hasattr(self, 'embedding') : 
            inputs_s_norm = self.embedding(inputs_s_norm)

        h_coords = self.spatial_net(inputs_s_norm)
        h_param = self.params_net(inputs_p_norm)

        combined = torch.cat([h_coords, h_param], dim=1)

        out = self.combined_net(combined)

        return out[:,0:1], out[:, 1:2]
