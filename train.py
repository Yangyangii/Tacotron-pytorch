from config import ConfigArgs as args
import os, sys, shutil
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter

import numpy as np
import pandas as pd
from collections import deque
from model import Tacotron
from data import SpeechDataset, collate_fn, load_vocab
from utils import att2img, spectrogram2wav, plot_att

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, data_loader, valid_loader, optimizer, scheduler, batch_size=32, ckpt_dir=None, writer=None, mode='1'):
    epochs = 0
    global_step = args.global_step
    criterion = nn.L1Loss().to(DEVICE) # default average
    model_infos = [('None', 10000.)]*5
    GO_frames = torch.zeros([batch_size, 1, args.n_mels*args.r]).to(DEVICE) # (N, Ty/r, n_mels)
    idx2char = load_vocab()[-1]
    while global_step < args.max_step:
        epoch_loss_mel = 0
        epoch_loss_mag = 0
        for step, (texts, mels, mags) in tqdm(enumerate(data_loader), total=len(data_loader), unit='B', ncols=70, leave=False):
            optimizer.zero_grad()
            
            texts, mels, mags = texts.to(DEVICE), mels.to(DEVICE), mags.to(DEVICE)
            prev_mels = torch.cat((GO_frames, mels[:, :-1, :]), 1)
            mels_hat, mags_hat, A = model(texts, prev_mels)

            loss_mel = criterion(mels_hat, mels)
            loss_mag = criterion(mags_hat, mags)
            loss = loss_mel + loss_mag
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scheduler.step()
            optimizer.step()
            
            epoch_loss_mel += loss_mel.item()
            epoch_loss_mag += loss_mag.item()
            global_step += 1
            if global_step % args.save_term == 0:
                model.eval() # 
                val_loss = evaluate(model, valid_loader, criterion, writer, global_step, args.test_batch)
                model_infos = save_model(model, model_infos, optimizer, scheduler, val_loss, global_step, ckpt_dir) # save best 5 models
                model.train()
        if args.log_mode:
            # Summary
            avg_loss_mel = epoch_loss_mel / (len(data_loader))
            avg_loss_mag = epoch_loss_mag / (len(data_loader))
            writer.add_scalar('train/loss_mel', avg_loss_mel, global_step)
            writer.add_scalar('train/loss_mag', avg_loss_mag, global_step)
            writer.add_scalar('train/lr', scheduler.get_lr()[0], global_step)
            
            alignment = A[0:1].clone().cpu().detach().numpy()
            writer.add_image('train/alignments', att2img(alignment), global_step) # (Tx, Ty)
            text = texts[0].cpu().detach().numpy()
            text = [idx2char[ch] for ch in text]
            plot_att(alignment[0], text, global_step, path=os.path.join(args.logdir, model.name, 'A', 'train'))
            
            mel_hat = mels_hat[0:1].transpose(1,2)
            mel = mels[0:1].transpose(1, 2)
            writer.add_image('train/mel_hat', mel_hat, global_step)
            writer.add_image('train/mel', mel, global_step)
            
            mag_hat = mags_hat[0:1].transpose(1, 2)
            mag = mags[0:1].transpose(1, 2)
            writer.add_image('train/mag_hat', mag_hat, global_step)
            writer.add_image('train/mag', mag, global_step)
            # print('Training Loss: {}'.format(avg_loss))
        epochs += 1
    print('Training complete')

def evaluate(model, data_loader, criterion, writer, global_step, batch_size=100):
    valid_loss_mel = 0.
    valid_loss_mag = 0.
    A = None 
    with torch.no_grad():
        for step, (texts, mels, mags) in enumerate(data_loader):
            texts, mels, mags = texts.to(DEVICE), mels.to(DEVICE), mags.to(DEVICE)
            GO_frames = torch.zeros([mels.shape[0], 1, args.n_mels*args.r]).to(DEVICE)  # (N, Ty/r, n_mels)
            prev_mels = torch.cat((GO_frames, mels[:, :-1, :]), 1)
            mels_hat, mags_hat, A = model(texts, prev_mels)

            loss_mel = criterion(mels_hat, mels)
            loss_mag = criterion(mags_hat, mags)
            valid_loss_mel += loss_mel.item()
            valid_loss_mag += loss_mag.item()
        avg_loss_mel = valid_loss_mel / (len(data_loader))
        avg_loss_mag = valid_loss_mag / (len(data_loader))
        writer.add_scalar('eval/loss_mel', avg_loss_mel, global_step)
        writer.add_scalar('eval/loss_mag', avg_loss_mag, global_step)

        alignment = A[0:1].clone().cpu().detach().numpy()
        writer.add_image('eval/alignments', att2img(alignment), global_step) # (Tx, Ty)
        text = texts[0].cpu().detach().numpy()
        text = [load_vocab()[-1][ch] for ch in text]
        plot_att(alignment[0], text, global_step, path=os.path.join(args.logdir, model.name, 'A'))

        mel_hat = mels_hat[0:1].transpose(1,2)
        mel = mels[0:1].transpose(1, 2)
        writer.add_image('eval/mel_hat', mel_hat, global_step)
        writer.add_image('eval/mel', mel, global_step)

        mag_hat = mags_hat[0:1].transpose(1, 2)
        mag = mags[0:1].transpose(1, 2)
        writer.add_image('eval/mag_hat', mag_hat, global_step)
        writer.add_image('eval/mag', mag, global_step)
    return avg_loss_mel

def save_model(model, model_infos, optimizer, scheduler, val_loss, global_step, ckpt_dir):
    cur_ckpt = 'model-{:03d}k.pth.tar'.format(global_step//1000)
    prev_ckpt = 'model-{:03d}k.pth.tar'.format(global_step//1000-(args.save_term//1000))
    state = {
        'global_step': global_step,
        'name': model.name,
        'model': model.state_dict(),
        'loss': val_loss,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }
    torch.save(state, os.path.join(ckpt_dir, cur_ckpt))
    if prev_ckpt not in dict(model_infos).keys() and os.path.exists(os.path.join(ckpt_dir, prev_ckpt)):
        os.remove(os.path.join(ckpt_dir, prev_ckpt))
    if val_loss < model_infos[-1][1]: # save better models
        worst_model = os.path.join(ckpt_dir, model_infos[-1][0])
        if os.path.exists(worst_model):
            os.remove(worst_model)
        model_infos[-1] = (cur_ckpt, float('{:.5f}'.format(val_loss)))
        model_infos = sorted(list(model_infos), key=lambda x: x[1])
        pd.DataFrame(model_infos).to_csv(os.path.join(ckpt_dir, 'ckpt.csv'), 
                                            sep=',', header=None, index=None)
    return model_infos

def main():
    model = Tacotron().to(DEVICE)
    print('Model {} is working...'.format(model.name))
    print('{} threads are used...'.format(torch.get_num_threads()))
    ckpt_dir = os.path.join(args.logdir, model.name)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.lr_decay_step//10, gamma=0.933) # around 1/2 per decay step

    if not os.path.exists(ckpt_dir):
        os.makedirs(os.path.join(ckpt_dir, 'A', 'train'))
    elif not os.path.exists(os.path.join(ckpt_dir, 'ckpt.csv')):
        shutil.rmtree(ckpt_dir)
        os.makedirs(os.path.join(ckpt_dir, 'A', 'train'))
    else:
        print('Already exists. Retrain the model.')
        ckpt = pd.read_csv(os.path.join(ckpt_dir, 'ckpt.csv'), sep=',', header=None)
        ckpt.columns = ['models', 'loss']
        ckpt = ckpt.sort_values(by='loss', ascending=True)
        state = torch.load(os.path.join(ckpt_dir, ckpt.models.loc[0]))
        model.load_state_dict(state['model'])
        args.global_step = state['global_step']
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])

    # model = torch.nn.DataParallel(model, device_ids=list(range(args.no_gpu))).to(DEVICE)
    
    dataset = SpeechDataset(args.data_path, args.meta_train, model.name, mem_mode=args.mem_mode)
    validset = SpeechDataset(args.data_path, args.meta_eval, model.name, mem_mode=args.mem_mode)
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                             shuffle=True, collate_fn=collate_fn,
                             drop_last=True, pin_memory=True)
    valid_loader = DataLoader(dataset=validset, batch_size=args.test_batch,
                              shuffle=False, collate_fn=collate_fn, pin_memory=True)
    
    writer = SummaryWriter(ckpt_dir)
    train(model, data_loader, valid_loader, optimizer, scheduler,
          batch_size=args.batch_size, ckpt_dir=ckpt_dir, writer=writer)
    return None

if __name__ == '__main__':
    # Set random seem for reproducibility
    seed = 999
    #manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == 'cuda':
        torch.cuda.manual_seed(seed)
    main()
