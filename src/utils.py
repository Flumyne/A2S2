import matplotlib.pyplot as plt   


def visualize_loss(loss_history, loss_pde_history, loss_bc_history):
    """
    Generates plots of training loss history to monitor convergence behavior.

    Args:
        Various history lists containing loss values per epoch.
    """

    fig_res, axes_res = plt.subplots(3, 1, figsize=(12, 8))

    axes_res[0].plot(loss_history)
    axes_res[0].set_title(f"Total Loss")
    axes_res[0].set_yscale('log')

    axes_res[1].plot(loss_pde_history)
    axes_res[1].set_title(f"PDE")
    axes_res[1].set_yscale('log')

    axes_res[2].plot(loss_bc_history)
    axes_res[2].set_title(f"BC")
    axes_res[2].set_yscale('log')


    plt.tight_layout()
    plt.savefig("Residual_A2S2_V0.png", dpi=150, bbox_inches='tight')