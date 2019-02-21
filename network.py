from config import ConfigArgs as args
import torch
import torch.nn as nn
import numpy as np
import module as mm

class PreNet(nn.Module):
    """
    PreNet
    Args
        x: (N, Tx, Ce) Character embedding (variable length)
    Returns:
        y_: (N, Tx, Cx)
    """

    def __init__(self, input_dim, hidden_dim):
        super(PreNet, self).__init__()
        self.prenet = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )

    def forward(self, x):
        y_ = self.prenet(x)
        return y_


class CBHG(nn.Module):
    """
    Convolution Bank HighyNet GRU
    Args:
        x: (N, Tx, input_dim) Tensor (variable length)
    Returns:
        y_: (N, hidden_dim, Tx) 
    """

    def __init__(self, input_dim, hidden_dim, K=16, n_highway=4):
        super(CBHG, self).__init__()
        self.K = K
        self.conv_bank = mm.Conv1dBank(input_dim, hidden_dim, K=self.K, activation_fn=torch.relu)
        self.max_pool = nn.MaxPool1d(2, stride=1, padding=1)
        self.projection = nn.Sequential(
            mm.Conv1d(self.K*hidden_dim, hidden_dim, 3, activation_fn=torch.relu, bias=False, bn=True),
            mm.Conv1d(hidden_dim, input_dim, 3, bias=False, bn=True),
        )
        self.highway = nn.ModuleList(
            [mm.Highway(input_dim) for _ in range(n_highway)]
        )
        self.gru = nn.GRU(input_dim, hidden_dim//2, num_layers=1, batch_first=True, bidirectional=True) # if batch_first is True, (Batch, Sequence, Feature)

    def forward(self, x, prev=None):
        y_ = x.transpose(1, 2) # (N, input_dim, Tx)
        y_ = self.conv_bank(y_) # (N, K*hidden_dim, Tx)
        y_ = self.max_pool(y_)[:, :, :-1] # pooling over time
        y_ = self.projection(y_) # (N, input_dim, Tx)
        y_ = y_.transpose(1, 2) # (N, Tx, input_dim)
        # Residual connection
        y_ = y_ + x  # (N, Tx, input_dim)
        for idx in range(len(self.highway)):
            y_ = self.highway[idx](y_)  # (N, Tx, input_dim)
        y_, hidden = self.gru(y_, prev)  # (N, Tx, hidden_dim)
        return y_, hidden


class ContextEncoder(nn.Module):
    """
    Text Encoder
        x: (N, Tx,) Text (variable length)
    Returns:
        y_: (N, Cx, Tx) Text Encoding for Key
    """
    def __init__(self):
        super(ContextEncoder, self).__init__()
        self.prenet = PreNet(args.Ce, args.Cx)
        self.cbhg = CBHG(input_dim=args.Cx, hidden_dim=args.Cx, K=16, n_highway=4)

    def forward(self, x):
        y_ = self.prenet(x)  # (N, Tx, Cx)
        y_, hidden = self.cbhg(y_)  # (N, Cx, Tx)
        return y_, hidden


class AudioDecoder(nn.Module):
    """
    Audio Decoder
    Args:
        decoder_inputs: (N, Ty/r, n_mels*r) Decoder inputs (previous decoder outputs)
        encoder_outputs: (N, Tx, Cx) Encoder outputs
    Returns:
        mel: (N, Ty/r, n_mels*r)
        mag: (N, Ty, n_mags)
        A: (N, Ty/r, Tx)
    """
    def __init__(self):
        super(AudioDecoder, self).__init__()
        self.prenet = PreNet(args.n_mels*args.r, args.Cx)
        self.cbhg = CBHG(input_dim=args.n_mels, hidden_dim=args.Cx, K=8, n_highway=4)
        self.attention_rnn = mm.AttentionRNN()
        self.proj_att = nn.Linear(args.Cx*2, args.Cx)
        self.decoder_rnn = nn.ModuleList([
            nn.GRU(args.Cx, args.Cx, num_layers=1, batch_first=True, bidirectional=False)
            for _ in range(2)
        ])
        self.proj_mel = nn.Linear(args.Cx, args.n_mels*args.r)
        self.proj_mag = nn.Linear(args.Cx, args.n_mags)

    def forward(self, decoder_inputs, encoder_outputs, prev_hidden=None, synth=False):
        if not synth: # Train mode & Eval mode
            y_ = self.prenet(decoder_inputs) # (N, Ty/r, Cx)
            # Attention RNN
            y_, A, hidden = self.attention_rnn(encoder_outputs, y_) # y_: (N, Ty/r, Cx), A: (N, Ty/r, Tx)
            # bmm (N, Ty/r, Tx) . (N, Tx, Cx)
            c = torch.bmm(A, encoder_outputs) # (N, Ty/r, Cx)
            y_ = self.proj_att(torch.cat([c, y_], dim=-1)) # (N, Ty/r, Cx)

            # Decoder RNN
            for idx in range(len(self.decoder_rnn)):
                y_f, _ = self.decoder_rnn[idx](y_)  # (N, Ty/r, Cx)
                y_ = y_ + y_f

            # Mel-spectrogram
            mels_hat = self.proj_mel(y_) # (N, Ty/r, n_mels*r)

            # Decoder CBHG
            y_ = mels_hat.view(mels_hat.size(0), -1, args.n_mels) # (N, Ty, n_mels)
            y_, _ = self.cbhg(y_)  # (N, Ty, Cx)
            mags_hat = self.proj_mag(y_) # (N, Ty, n_mags)
            return mels_hat, mags_hat, A
        else:
            # decoder_inputs: GO frame (N, 1, n_mels*r)
            att_hidden = None
            dec_hidden = [None, None]

            mels_hat = []
            mags_hat = []
            attention = []
            for idx in range(args.max_Ty):
                y_ = self.prenet(decoder_inputs)  # (N, 1, Cx)
                # Attention RNN
                y_, A, att_hidden = self.attention_rnn(encoder_outputs, y_, prev_hidden=att_hidden)
                attention.append(A)
                # Encoder outputs: (N, Tx, Cx)
                # A: (N, )
                c = torch.bmm(A, encoder_outputs)  # (N, Ty/r, Cx)
                y_ = self.proj_att(torch.cat([c, y_], dim=-1))  # (N, Ty/r, Cx)

                # Decoder RNN
                for j in range(len(self.decoder_rnn)):
                    y_f, dec_hidden[j] = self.decoder_rnn[j](y_, dec_hidden[j])  # (N, 1, Cx)
                    y_ = y_ + y_f  # (N, 1, Cx)

                # Mel-spectrogram
                mel_hat = self.proj_mel(y_)  # (N, 1, n_mels*r)
                decoder_inputs = mel_hat[:, :, -args.n_mels*args.r:] # ?????
                mels_hat.append(mel_hat)
            
            mels_hat = torch.cat(mels_hat, dim=1)
            attention = torch.cat(attention, dim=1)

            # Decoder CBHG
            y_ = mels_hat.view(mels_hat.size(0), -1, args.n_mels) # (N, Ty, n_mels)
            y_, _ = self.cbhg(y_)
            mags_hat = self.proj_mag(y_)
            
            return mels_hat, mags_hat, attention
