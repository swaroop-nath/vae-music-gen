import argparse
import torch
from vae import VAE
import pypianoroll
from config_manager import ConfigManager
from data_handler import load_gen_data
from random import randint
from IPython.display import Audio
import os
from manager import DEVICE
from torch.nn.functional import relu

def gen_sample(vae, sample, gen_len):
    generated_track = torch.zeros((1, 32 * (gen_len+1), 128))
    generated_track[0, 0:32, :] = sample * 127

    for i in range(1, gen_len+1):
        sample = sample.to(DEVICE)
        sample = relu(vae.generate_music_roll(sample).detach().to('cpu'))
        # print(sample)
        # exit(1)
        generated_track[0, 32*i:32*(i+1), :] = sample * 127 * 2

    piano_track = pypianoroll.StandardTrack(name = 'Piano', program = 0, is_drum = False, pianoroll = generated_track[0, :, :])
    generated_multitrack = pypianoroll.Multitrack(name = 'Generated', resolution = 2, tracks = [piano_track])
    generated_midi = pypianoroll.to_pretty_midi(generated_multitrack)
    generated_midi_audio = generated_midi.fluidsynth()
    
    return generated_midi_audio, generated_multitrack

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generation script')
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--latent_dim', type=int, default=10)
    parser.add_argument('--saved_path', type=str, required=True)
    args = parser.parse_args()

    cfg_mgr = ConfigManager()
    kwargs = cfg_mgr.get_kwargs()

    num_gen = 1
    gen_len = 10
    vae = VAE(num_instruments=1, temporal_unit_size=32, num_pitches=128, latent_dim=args.latent_dim, **kwargs)
    vae.to(DEVICE)
    vae.load_state_dict(torch.load(args.saved_path))

    pianorolls = load_gen_data(args.root_dir)
    num_rolls = len(pianorolls)

    for _ in range(num_gen):
        start_idx = 1 # randint(0, num_rolls-gen_len-1)
        sample = pianorolls[start_idx]
        actual_track = torch.zeros((1, 32 * (gen_len+1), 128))
        actual_track[0, 0:32, :] = sample * 127

        for i in range(1, gen_len+1):
            actual_track[0, 32*i:32*(i+1), :] = pianorolls[start_idx+i] * 127

        actual_piano_track = pypianoroll.StandardTrack(name = 'Piano', program = 0, is_drum = False, pianoroll = actual_track[0, :, :])
        actual_multitrack = pypianoroll.Multitrack(name = 'Generated', resolution = 2, tracks = [actual_piano_track])
        actual_midi = pypianoroll.to_pretty_midi(actual_multitrack)
        actual_midi_audio = actual_midi.fluidsynth()

        generated_midi_audio, generated_multitrack = gen_sample(vae, sample, gen_len)

        print('Actual audio - \n')
        Audio(actual_midi_audio, rate=44100)
        print('Generated audio - \n')
        Audio(generated_midi_audio, rate=44100)

        if not os.path.isdir('results'): os.makedirs('results')
        actual_path = './results/actual_{}.mid'.format(_)
        gen_path = './results/gen_{}.mid'.format(_)
        
        pypianoroll.write(gen_path, generated_multitrack)
        pypianoroll.write(actual_path, actual_multitrack)

