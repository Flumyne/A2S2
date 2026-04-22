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
    def __init__(self, normalizer, input_dim=2, hidden_dim=32, output_dim=2, num_layers=5, use_fourier=True):
        super(NeuralNet, self).__init__()

        #self.register_buffer('mu', torch.zeros(input_dim))
        #self.register_buffer('sigma', torch.ones(input_dim))
        self.normalizer = normalizer
        
        self.use_fourier = use_fourier
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
        
        self.net = nn.Sequential(*layers)

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

        raw = self.net(inputs)
        mask = x
        u = mask * raw[:,0:1]
        v = mask * raw[:,1:2]

        return torch.cat([u,v], dim=1)

if __name__ == "__main__":
    # Test unitaire rapide
    model = NeuralNet(input_dim=2, hidden_dim=32, output_dim=2)
    x = torch.linspace(0, 1, 10, requires_grad=True).view(-1, 1)
    y = torch.linspace(0, 1, 10, requires_grad=True).view(-1, 1)
    
    outputs = model(x, y)
    print(f"Output shape: {outputs.shape}") # [10, 2] -> (u, v)
