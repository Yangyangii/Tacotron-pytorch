from config import ConfigArgs as args
import torch
import torch.nn as nn
from network import ContextEncoder, AudioDecoder
import module as mm

class Tacotron(nn.Module):
    """
    Text2Mel
    Args:
        L: (N, Tx) text
        S: (N, Ty/r, n_mels*r) previous audio
    Returns:
        mels_hat: (N, Ty/r, n_mels*r)
        mags_hat: (N, Ty, n_mags)
        attentions: (N, Ty/r, Tx)
    """
    def __init__(self):
        super(Tacotron, self).__init__()
        self.name = 'Tacotron'
        self.embed = nn.Embedding(len(args.vocab), args.Ce, padding_idx=0)
        self.encoder = ContextEncoder()
        self.decoder = AudioDecoder()
    
    def forward(self, enc_inputs, dec_inputs, synth=False):
        x = self.embed(enc_inputs)  # (N, Tx, Ce)
        enc_outputs, enc_hidden = self.encoder(x)
        mels_hat, mags_hat, attentions = self.decoder(dec_inputs, enc_outputs, synth=synth)
        return mels_hat, mags_hat, attentions
