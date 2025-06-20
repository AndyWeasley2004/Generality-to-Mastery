import os
import pickle
import sys
import time
import shutil
from turtle import Turtle
import yaml
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast
from torch.amp.grad_scaler import GradScaler
from dataloader import MusicSegmentDataset
from model.vanilla_encoder import ComposerClassifier
scaler = GradScaler(device='cuda')

def get_module_gradient_norm(module, norm_type=2):
    total_norm = 0.0
    for param in module.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

def build_composer_mapping(used_categories):
    return {cat.lower(): idx for idx, cat in enumerate(used_categories)}

def train(epoch, model, dataloader, optimizer, scheduler, composer2idx, device='cuda'):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    for idx, batch in enumerate(dataloader):
        inp = batch['dec_inp'][:, 1:]
        inp = inp.to(device)
        
        composers = batch['composer']
        target = torch.LongTensor([composer2idx[c.split('_')[1].lower()] for c in composers]).to(device)

        optimizer.zero_grad()
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(inp)  # (batch_size, num_classes)
            loss = F.cross_entropy(logits, target)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = get_module_gradient_norm(model.encoder.layers[0])
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() * inp.size(0)
        preds = logits.argmax(dim=-1)
        total_correct += (preds == target).sum().item()
        total_samples += inp.size(0)
        
        sys.stdout.write('\r batch {:03d}: loss: {:.4f}, accuracy: {:.4f}, grad: {:.3f}'.format(
                idx,
                total_loss / total_samples,
                100*total_correct / total_samples,
                grad_norm
            ))
        sys.stdout.flush()

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    print(f"\r[Train Epoch {epoch}] Loss: {avg_loss:.4f}, Accuracy: {accuracy*100:.2f}%")
    return avg_loss, accuracy

def validate(epoch, model, dataloader, composer2idx, device='cuda'):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            inp = batch['dec_inp'][:, 1:]
            inp = inp.to(device)
            composers = batch['composer']
            target = torch.LongTensor([composer2idx[c.split('_')[1].lower()] for c in composers]).to(device)
            
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(inp)
                loss = F.cross_entropy(logits, target)
            total_loss += loss.item() * inp.size(0)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == target).sum().item()
            total_samples += inp.size(0)
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    print(f"[Val Epoch {epoch}] Loss: {avg_loss:.4f}, Accuracy: {accuracy*100:.2f}%")
    return avg_loss, accuracy


# Main training script
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Composer Classification Training')
    required = parser.add_argument_group('required arguments')
    required.add_argument('-c', '--configuration', help='Training configuration', required=True)
    args = parser.parse_args()

    config_path = args.configuration
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    ckpt_dir = config['output']['ckpt_dir']
    os.makedirs(ckpt_dir, exist_ok=True)
    device = config['device']
    torch.cuda.set_device(device)

    # Composer mapping based on used categories.
    used_categories = ['bach', 'chopin', 'mozart', 'beethoven']
    composer2idx = build_composer_mapping(used_categories)
    num_classes = len(composer2idx)

    # Initialize datasets.
    train_dset = MusicSegmentDataset(
        vocab_path=config['data']['vocab_path'],
        data_dir=config['data']['data_dir'],
        model_dec_seqlen=config['model']['encoder']['tgt_len'],
        augment=True,
        from_start_only=config['data']['from_start_only'],
        split='train',
        train_ratio=0.9,
        used_categories=used_categories,
        epoch_steps=config['data'].get('epoch_steps', 1000),
        eval=False
    )
    val_dset = MusicSegmentDataset(
        vocab_path=config['data']['vocab_path'],
        data_dir=config['data']['data_dir'],
        model_dec_seqlen=config['model']['encoder']['tgt_len'],
        augment=False,
        from_start_only=True,
        split='valid',
        train_ratio=0.9,
        used_categories=used_categories
    )

    print('[Dataset sizes]', len(train_dset), len(val_dset))
    dloader = DataLoader(
        train_dset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=4,
        collate_fn=train_dset.collate_fn
    )
    val_dloader = DataLoader(
        val_dset,
        batch_size=config['data']['batch_size'],
        num_workers=4,
        collate_fn=val_dset.collate_fn
    )

    # Create the ComposerClassifier model.
    mconf = config['model']['encoder']
    event2idx, _ = pickle.load(open(config['data']['vocab_path'], 'rb'))
    model = ComposerClassifier(
        d_word_embed=mconf['d_model'],
        event2word=event2idx,
        n_layer=mconf['n_layer'],
        n_head=mconf['n_head'],
        d_model=mconf['d_model'],
        d_ff=mconf['d_ff'],
        max_seq_len=mconf['tgt_len'],
        num_classes=num_classes,
        dropout=mconf['dropout'],
        pre_lnorm=config['model']['pre_lnorm']
    ).to(device)

    print('[Info] # params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    # Setup optimizer and scheduler.
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=config['training']['max_lr'],
                           weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['lr_decay_steps'],
        eta_min=config['training']['min_lr']
    )

    optim_ckpt_path = config['pretrained_optim_path']
    param_ckpt_path = config['pretrained_param_path']
    if optim_ckpt_path:
        optimizer.load_state_dict(torch.load(optim_ckpt_path, map_location=device, weights_only=True))
    if param_ckpt_path:
        model.load_state_dict(torch.load(param_ckpt_path, map_location=device, weights_only=True), strict=False)
        print(f'Loading weight from {param_ckpt_path}')

    log_interval = config['training']['log_interval']
    start_epoch = config['training'].get('trained_epochs', 0)
    max_epoch = config['training']['max_epoch']
    log_file = os.path.join(ckpt_dir, 'log.txt') if start_epoch == 0 else os.path.join(ckpt_dir, f'log_from_ep{start_epoch:03d}.txt')

    # Main training loop.
    for ep in range(start_epoch, max_epoch):
        t_loss, t_acc = train(ep + 1, model, dloader, optimizer, scheduler, composer2idx, device=device)
        if (ep + 1) % config['output']['ckpt_interval'] == 0:
            ckpt_path = os.path.join(ckpt_dir, f'ep{ep+1:03d}_loss{t_loss:.3f}_params.pt')
            torch.save(model.state_dict(), ckpt_path)
        
        if (ep + 1) % config['training']['val_interval'] == 0:
            v_loss, v_acc = validate(ep + 1, model, val_dloader, composer2idx, device=device)
            with open(log_file, 'a') as f:
                f.write(f"Ep {ep+1:03d} | Train Loss: {t_loss:.3f} | Val Loss: {v_loss:.3f} | "
                        f"Train Acc: {t_acc*100:.2f}% | Val Acc: {v_acc*100:.2f}%\n")
        print(f"[Epoch {ep+1:03d}] completed. Train loss: {t_loss:.4f}, Train acc: {t_acc*100:.2f}%")
