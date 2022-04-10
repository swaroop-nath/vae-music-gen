from torch.optim import AdamW
import torch
from tqdm import tqdm
from config_manager import ConfigManager
from manager import DEVICE
import numpy as np
from data_handler import load_data
import argparse
import os
from vae import VAE

class VAETrainer:
    def __init__(self, num_epochs, vae, optim, data_loader, save_dir, test_split):
        self.num_epochs = num_epochs
        self.vae = vae
        self.optim = optim
        self.data_loader = data_loader
        if not os.path.isdir(save_dir): os.makedirs(save_dir)
        self.save_path = save_dir + '/saved_model.pt'
        self.train_limit = int(np.ceil((1-test_split) * len(self.data_loader)))
        self.test_limit = len(self.data_loader) - self.train_limit

    def train(self):
        for epoch in range(1, self.num_epochs + 1):
            avg_batch_elbo_train, avg_batch_elbo_valid = 0, 0

            p_bar_train = tqdm(total=len(range(self.train_limit)), desc='Training on batches')
            p_bar_valid = None
            batch_idx = 0

            for batch in self.data_loader:
                batch = batch.to(DEVICE)

                if batch_idx < self.train_limit:
                    self.optim.zero_grad()
                    _, loss = self.vae(batch)
                    loss.backward()
                    self.optim.step()

                    loss_item = loss.clone().detach().item()
                    avg_batch_elbo_train += -loss_item / self.train_limit

                    p_bar_train.set_postfix(ELBO=-loss_item)
                    p_bar_train.update(1)

                else:
                    if batch_idx == self.train_limit:
                        p_bar_train.close()
                        print('Avg training ELBO after epoch {}: {:0.4f}\n'.format(epoch, avg_batch_elbo_train))
                        p_bar_valid = tqdm(total=len(range(self.test_limit)), desc='Evaluation on validation set')

                    with torch.no_grad():
                        _, loss = self.vae(batch)

                    loss_item = loss.clone().detach().item()
                    avg_batch_elbo_valid += -loss_item / self.test_limit

                    p_bar_valid.set_postfix(ELBO=-loss_item)
                    p_bar_valid.update(1)

                batch_idx += 1

            p_bar_valid.close()
            print('Avg validation ELBO after epoch {}: {:0.4f}\n'.format(epoch, avg_batch_elbo_valid))

        print('Learned sigma: {:0.5f}'.format(self.vae.log_sig_x.detach().item()))
        self._save_model()

    def _save_model(self):
        torch.save(self.vae.state_dict(), self.save_path)
        print('Saved model at ' + self.save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run training')
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--latent_dim', type=int, default=10)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--test_split', type=float, default=0.2)
    args = parser.parse_args()

    cfg_mgr = ConfigManager()
    kwargs = cfg_mgr.get_kwargs()

    if not os.path.isdir(args.root_dir): os.makedirs(args.root_dir)

    data_loader = load_data(root_dir=args.root_dir, batch_size=args.batch_size)
    vae = VAE(num_instruments=1, temporal_unit_size=32, num_pitches=128, latent_dim=args.latent_dim, **kwargs)
    vae.to(DEVICE)
    optim = AdamW(params=vae.parameters(), lr=args.learning_rate)

    trainer = VAETrainer(args.num_epochs, vae, optim, data_loader, args.save_dir, args.test_split)
    trainer.train()