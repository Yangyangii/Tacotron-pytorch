from config import ConfigArgs as args
import os, sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

import numpy as np
import pandas as pd
from model import Tacotron
from data import TextDataset, synth_collate_fn, load_vocab
import utils
from scipy.io.wavfile import write
import glob

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def synthesize(model, data_loader, batch_size=100):
    '''
    Tacotron

    '''
    idx2char = load_vocab()[-1]
    with torch.no_grad():
        print('*'*15, ' Synthesize ', '*'*15)
        mags = torch.zeros([len(data_loader.dataset), args.max_Ty*args.r, args.n_mags]).to(DEVICE)
        for step, (texts, _, _) in enumerate(data_loader):
            texts = texts.to(DEVICE)
            GO_frames = torch.zeros([texts.shape[0], 1, args.n_mels*args.r]).to(DEVICE)
            _, mags_hat, A = model(texts, GO_frames, synth=True)
            
            print('='*10, ' Alignment ', '='*10)
            alignments = A.cpu().detach().numpy()
            visual_texts = texts.cpu().detach().numpy()
            for idx in range(len(alignments)):
                text = [idx2char[ch] for ch in visual_texts[idx]]
                utils.plot_att(alignments[idx], text, args.global_step, path=os.path.join(args.sampledir, 'A'), name='{}.png'.format(idx))
            mags[step*batch_size:(step+1)*batch_size:, :, :] = mags_hat # mag: (N, Ty, n_mags)
        print('='*10, ' Vocoder ', '='*10)
        mags = mags.cpu().detach().numpy()
        for idx in trange(len(mags), unit='B', ncols=70):
            wav = utils.spectrogram2wav(mags[idx])
            write(os.path.join(args.sampledir, '{}.wav'.format(idx+1)), args.sr, wav)
    return None

def main():
    testset = TextDataset(args.testset)
    test_loader = DataLoader(dataset=testset, batch_size=args.test_batch, drop_last=False,
                             shuffle=False, collate_fn=synth_collate_fn, pin_memory=True)

    model = Tacotron().to(DEVICE)
    
    model_path = sorted(glob.glob(os.path.join(args.logdir, model.name, 'model-*.tar')))[-1] # latest model
    state = torch.load(model_path)
    model.load_state_dict(state['model'])
    args.global_step = state['global_step']

    print('The model is loaded. Step: {}'.format(args.global_step))

    model.eval()
    
    if not os.path.exists(os.path.join(args.sampledir, 'A')):
        os.makedirs(os.path.join(args.sampledir, 'A'))
    synthesize(model, test_loader, args.test_batch)

if __name__ == '__main__':
    main()
