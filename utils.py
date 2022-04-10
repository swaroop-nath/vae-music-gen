import torch
from torch.nn.functional import mse_loss

def kl_divergence(mu, sigma):
    # Returns the KL-divergence of N(μ, Σ) and N(0, I)
    # mu.size and sigma.size == (batch_size, latent_dim)
    kl_div = 0.5 * (torch.sum(mu * mu, dim=-1) + torch.sum(sigma, dim=-1) - mu.size(-1) - torch.log(torch.abs(torch.prod(sigma, dim=-1))))
    kl_div = kl_div.mean()
    with open('kl_div.log', 'a') as file:
        file.write('KL Divergence: {:0.4f}\n'.format(kl_div.clone().detach().item()))
    return kl_div

# def log_p_x(actual, pred, sigma):
#     # actual.size and pred.size == (batch_size, temporal_unit_size, num_pitches)

#     loss = (pred - actual)**2 / (2 * sigma**2) + torch.log(sigma)
#     loss = -loss.sum(dim=-1).mean()

#     with open('reconstruction_loss.log', 'a') as file:
#         file.write('Reconstruction Loss: {:0.4f}\n'.format(loss.clone().detach().item()))
#     return loss

def log_p_x(x, mu_xs, sig_x):
    """Given [batch, ...] input x and [batch, n, ...] reconstructions, compute
    pixel-wise log Gaussian probability

    Sum over pixel dimensions, but mean over batch and samples.
    """
    # print(mu_xs.size(), x.size())
    b, n = mu_xs.size()[:2]
    # Flatten out pixels and add a singleton dimension [1] so that x will be
    # implicitly expanded when combined with mu_xs
    # x = x.reshape(b, 1, -1)
    _, _, p = x.size()
    squared_error = (x - mu_xs)**2 / (2*sig_x**2)
    return -(squared_error + torch.log(sig_x)).sum(dim=2).mean(dim=(0,1))

def empirical_kl_divergence(sampled_z, mu, sigma):
    # Returns the KL-divergence of N(μ, Σ) and N(0, I)
    # mu.size and sigma.size == (batch_size, latent_dim)
    # sampled_z.size == (batch_size, 1, latent_dim)
    sampled_z = sampled_z.squeeze(1)
    log_p = -0.5 * (sampled_z ** 2) # standard Gaussian density = exp(-z^2/2)
    log_q = torch.div(-0.5 * ((sampled_z - mu)**2), sigma) - torch.log(sigma)

    kl_div = (log_q - log_p).sum(dim=-1).mean()

    with open('kl_div.log', 'a') as file:
        file.write('KL Divergence: {:0.4f}\n'.format(kl_div.clone().detach().item()))
    return kl_div