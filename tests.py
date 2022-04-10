from vae import VAE
import torch
from config_manager import ConfigManager

def test_vae(**kwargs):
    vae = VAE(num_instruments=1, temporal_unit_size=32, num_pitches=128, latent_dim=2, filter_size=2, **kwargs)
    demo_batch = torch.randn((32, 32, 128))
    gen_notes, elbo = vae(demo_batch)

    assert gen_notes.size() == demo_batch.size()
    assert elbo.size() == torch.Size((32, 1))

if __name__ == '__main__':
    cfg_mgr = ConfigManager()
    kwargs = cfg_mgr.get_kwargs()

    test_vae(**kwargs)