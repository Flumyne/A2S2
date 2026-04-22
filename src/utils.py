import matplotlib.pyplot as plt  
import torch 

class Normalizer:
    """
    A utility class to normalize and denormalize tensors using mean and standard deviation statistics.
    """
    def __init__(self, tensor=None, mean=None, std=None, device='cpu'):
        super(Normalizer,self).__init__()

        if tensor is not None:
            self.mean = torch.mean(tensor, dim=0).detach().to(device)
            self.std = torch.std(tensor, dim=0).detach().to(device)
            self.std[self.std < 1e-6] = 1.0
        else:
            self.mean = mean.to(device)
            self.std = std.to(device)
        
    def encode(self, x):
        """
        Normalizes the input tensor.

        Args:
            x (torch.Tensor): The raw input tensor.

        Returns:
            torch.Tensor: The normalized tensor.
        """
        return (x - self.mean) / self.std    
    
    def decode(self, x):
        """
        Denormalizes the input tensor back to its original scale.

        Args:
            x (torch.Tensor): The normalized input tensor.

        Returns:
            torch.Tensor: The denormalized tensor.
        """
        return x * self.std + self.mean    

    def cuda(self):
        """
        Moves the normalizer statistics to the GPU.

        Returns:
            Normalizer: The normalizer instance on CUDA.
        """
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()
        return self    

def visualize_loss(loss_history, loss_energy_history, loss_bc_left_history):
    """
    Generates plots of training loss history to monitor convergence behavior.

    Args:
        Various history lists containing loss values per epoch.
    """

    fig_res, axes_res = plt.subplots(3, 1, figsize=(12, 8))

    axes_res[0].plot(loss_history)
    axes_res[0].set_title(f"Total Loss")
    axes_res.set_yscale('log')

    axes_res[1].plot(loss_energy_history)
    axes_res[1].set_title(f"Energy Loss")

    axes_res[2].plot(loss_bc_left_history)
    axes_res[2].set_title(f"Dirchlet Loss")
    axes_res.set_yscale('log')


    plt.tight_layout()
    plt.savefig("Residual_A2S2_V0_2.png", dpi=150, bbox_inches='tight')
    