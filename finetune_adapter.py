import os
import sys
import shutil
import time
import yaml
import torch
import argparse
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from torch.amp import autocast
from torch.amp.grad_scaler import GradScaler

from model.vanilla_transformer import VanillaTransformer
from dataloader import MusicSegmentDataset
from utils import pickle_load, compute_accuracy

sys.path.append('./model/')
scaler = GradScaler(device='cuda')

def get_module_gradient_norm(module, norm_type=2):
    total_norm = 0.0
    for param in module.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

def train(epoch, model, dloader, optim, sched, pad_token):
    model.train()
    recons_loss_rec = 0.
    accum_samples = 0

    print('[epoch {:03d}] training ...'.format(epoch))
    st = time.time()

    for batch_idx, batch_samples in enumerate(dloader):
        dec_input = batch_samples['dec_inp'].permute(1, 0).cuda()
        dec_target = batch_samples['dec_tgt'].permute(1, 0).cuda()
        inp_chord = batch_samples['inp_chord']
        inp_melody = batch_samples['inp_melody']

        global train_steps
        train_steps += 1

        with autocast(device_type='cuda', dtype=torch.bfloat16):
            dec_logits = model(dec_input)
            losses = model.compute_loss(dec_logits, dec_target)

        # clip gradient & update model
        if accum_steps > 1:
            losses /= accum_steps
        scaler.scale(losses).backward()

        if (train_steps % accum_steps) == 0:
            scaler.unscale_(optim)
            grad_norm = get_module_gradient_norm(model.decoder.adapters['3'])

            found_nonfinite = False
            for param in model.parameters():
                if param.grad is not None:
                    if not torch.all(torch.isfinite(param.grad)):
                        found_nonfinite = True
                        break

            if found_nonfinite:
                print("Warning: Non-finite gradients detected. Skipping update for this batch.")
                scaler.update()
                optim.zero_grad()
                continue
            else:
                adapter_params = [p for n, p in model.named_parameters() if "adapters" in n]
                torch.nn.utils.clip_grad_norm_(adapter_params, 2.5)
                other_params = [p for n, p in model.named_parameters() if "adapters" not in n]
                torch.nn.utils.clip_grad_norm_(other_params, 0.5)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()

            total_acc, chord_acc, melody_acc, others_acc = \
                compute_accuracy(dec_logits.cpu(), dec_target.cpu(), inp_chord, inp_melody, pad_token)

            recons_loss_rec += batch_samples['id'].size(0) * losses.item() * accum_steps * accum_steps
            accum_samples += batch_samples['id'].size(0) * accum_steps

            sys.stdout.write('\r batch {:03d}: loss: {:.4f}, total_acc: {:.4f}, '
                'gradient: {:.4f}, melody_acc: {:.4f}, other_acc: {:.4f}, '
                'step: {}, time_elapsed: {:.2f} secs'.format(
                batch_idx,
                recons_loss_rec / accum_samples,
                total_acc,
                grad_norm,
                melody_acc,
                others_acc,
                train_steps,
                time.time() - st
            ))
            sys.stdout.flush()

        # anneal learning rate
        if (train_steps // accum_steps) < warmup_steps:
            curr_lr = max_lr * train_steps / (warmup_steps * accum_steps)
            optim.param_groups[0]['lr'] = curr_lr
        else:
            sched.step()

        if not train_steps % log_interval:
            log_data = {
                'ep': epoch,
                'steps': train_steps,
                'ce_loss': recons_loss_rec / accum_samples,
                'total': total_acc,
                'chord': chord_acc,
                'melody': melody_acc,
                'others': others_acc,
                'time': time.time() - st
            }
            log_epoch(
                os.path.join(ckpt_dir, log_file), log_data,
                is_init=not os.path.exists(os.path.join(ckpt_dir, log_file))
            )

    return recons_loss_rec / accum_samples, time.time() - st


def validate(epoch, model, dloader, pad_token, rounds=1):
    model.eval()
    recons_loss_rec = []
    total_acc_rec = []
    chord_acc_rec = []
    melody_acc_rec = []
    others_acc_rec = []

    print('[epoch {:03d}] validating ...'.format(epoch))
    with torch.no_grad():
        for batch_idx, batch_samples in enumerate(dloader):
            dec_input = batch_samples['dec_inp'].permute(1, 0).cuda()
            dec_target = batch_samples['dec_tgt'].permute(1, 0).cuda()
            inp_chord = batch_samples['inp_chord']
            inp_melody = batch_samples['inp_melody']
            
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                dec_logits = model(dec_input)
                losses = model.compute_loss(dec_logits, dec_target)

            total_acc, chord_acc, melody_acc, others_acc = \
                compute_accuracy(dec_logits.cpu(), dec_target.cpu(), inp_chord, inp_melody, pad_token)

            recons_loss_rec.append(losses.item())
            total_acc_rec.append(total_acc)
            chord_acc_rec.append(chord_acc)
            melody_acc_rec.append(melody_acc)
            others_acc_rec.append(others_acc)

    return recons_loss_rec, total_acc_rec, chord_acc_rec, melody_acc_rec, others_acc_rec


def log_epoch(log_file, log_data, is_init=False):
    if is_init:
        with open(log_file, 'w') as f:
            f.write('{:4} {:8} {:12} {:12} {:12}\n'.format(
                'ep', 'steps', 'ce_loss', 'ep_time', 'total_time'
            ))

    with open(log_file, 'a') as f:
        f.write('{:<4} {:<8} {:<12} {:<12} {:<12}\n'.format(
            log_data['ep'],
            log_data['steps'],
            round(log_data['ce_loss'], 5),
            round(log_data['time'], 2),
            round(time.time() - init_time, 2)
        ))

    return


def compute_accuracy(dec_logits, dec_target, inp_chord, inp_melody, pad_token):
    dec_pred = torch.argmax(dec_logits, dim=-1).permute(1, 0).cpu().numpy()
    dec_target = dec_target.permute(1, 0).cpu().numpy()
    
    valid = dec_target != pad_token
    total_acc = np.mean((dec_pred[valid] == dec_target[valid]).astype(np.float32))
    
    chord_idx = (inp_chord.cpu().numpy() == 1)
    if np.sum(chord_idx) > 0:
        chord_acc = np.mean((dec_pred[chord_idx] == dec_target[chord_idx]).astype(np.float32))
    else:
        chord_acc = 0.0

    melody_idx = (inp_melody.cpu().numpy() == 1)
    if np.sum(melody_idx) > 0:
        melody_acc = np.mean((dec_pred[melody_idx] == dec_target[melody_idx]).astype(np.float32))
    else:
        melody_acc = 0.0

    total_count = np.sum(valid)
    chord_count = np.sum(chord_idx)
    melody_count = np.sum(melody_idx)
    others_count = total_count - chord_count - melody_count
    if others_count > 0:
        others_acc = (
            total_acc * total_count - chord_acc * chord_count - melody_acc * melody_count
        ) / others_count
    else:
        others_acc = 0.0

    return total_acc, chord_acc, melody_acc, others_acc


if __name__ == '__main__':
    # configuration
    parser = argparse.ArgumentParser(description='')
    required = parser.add_argument_group('required arguments')
    required.add_argument('-c', '--configuration',
                          help='configurations of training', required=True)
    args = parser.parse_args()

    config_path = args.configuration
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    ckpt_dir = config['output']['ckpt_dir']

    torch.cuda.set_device(config['device'])
    train_steps = 0
    start_epoch = 0 if config['training']['trained_epochs'] is None else config['training']['trained_epochs']
    warmup_steps = config['training']['warmup_steps']
    log_interval = config['training']['log_interval']
    max_lr = config['training']['max_lr']
    log_file = 'log.txt' if start_epoch == 0 else 'log_from_ep{:03d}.txt'.format(start_epoch)
    accum_steps = config['training']["accum_steps"] 
    optim_ckpt_path = config['pretrained_optim_path']
    param_ckpt_path = config['pretrained_param_path']

    init_time = time.time()

    params_dir = os.path.join(ckpt_dir, 'params/') if start_epoch == 0 \
        else os.path.join(ckpt_dir, 'params_from_ep{:03d}/'.format(start_epoch))
    optimizer_dir = os.path.join(ckpt_dir, 'optim/') if start_epoch == 0 \
        else os.path.join(ckpt_dir, 'optim_from_ep{:03d}/'.format(start_epoch))

    dset = MusicSegmentDataset(
        vocab_path=config['data']['vocab_path'],
        data_dir=config['data']['data_dir'],
        model_dec_seqlen=config['model']['decoder']['tgt_len'],
        augment=True,
        from_start_only=config['data']['from_start_only'],
        split='train',
        train_ratio=0.97,
        used_categories=['bach', 'chopin', 'schumann', 'mozart', 'beethoven'],
        epoch_steps=1000,
        strong_augment=True
    )

    val_dset = MusicSegmentDataset(
        vocab_path=config['data']['vocab_path'],
        data_dir=config['data']['data_dir'],
        model_dec_seqlen=config['model']['decoder']['tgt_len'],
        augment=False,
        from_start_only=True,
        split='valid',
        train_ratio=0.97,
        used_categories=['bach', 'chopin', 'schumann', 'mozart', 'beethoven']
    )

    event2idx, _ = pickle_load(config['data']['vocab_path'])
    print('[dset lens]', len(dset), len(val_dset))
    print('[vocab path]', config['data']['vocab_path'])
    print('[vocab size]', len(event2idx))

    dloader = DataLoader(
        dset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=4,
        collate_fn=dset.collate_fn
    )
    val_dloader = DataLoader(
        val_dset,
        batch_size=config['data']['batch_size']*2,
        shuffle=False,
        num_workers=4,
        collate_fn=val_dset.collate_fn
    )

    mconf = config['model']
    model = VanillaTransformer(
        mconf['d_word_embed'],
        event2idx,
        mconf['decoder']['n_layer'],
        mconf['decoder']['n_head'],
        mconf['decoder']['d_model'],
        mconf['decoder']['d_ff'],
        attn_type=mconf['attn_type'],
        tgt_len=mconf['decoder']['tgt_len'],
        dec_dropout=mconf['decoder']['dropout'],
        pre_lnorm=mconf['pre_lnorm'],
        adapter_positions=mconf['adapter_positions']
    ).cuda()
    print('[info] # params:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    opt_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(opt_params, lr=config['training']['max_lr'], weight_decay=0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['lr_decay_steps'],
        eta_min=config['training']['min_lr']
    )

    if optim_ckpt_path:
        optimizer.load_state_dict(
            torch.load(optim_ckpt_path, map_location=config['device'], weights_only=True)
        )

    if param_ckpt_path:
        model.load_state_dict(
            torch.load(param_ckpt_path, map_location=config['device'], weights_only=True), strict=False
        )
        print(f'Loading weight from {param_ckpt_path}')

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(params_dir, exist_ok=True)
    os.makedirs(optimizer_dir, exist_ok=True)
    shutil.copy(config_path, os.path.join(ckpt_dir, 'config.yaml'))

    for ep in range(start_epoch, config['training']['max_epoch']):
        recons_loss, ep_time = train(ep + 1, model, dloader, optimizer, scheduler, dset.pad_token)
        if not (ep + 1) % config['output']['ckpt_interval']:
            torch.save(model.state_dict(),
                       os.path.join(params_dir, 'ep{:03d}_loss{:.3f}_params.pt'.format(ep + 1, recons_loss))
                       )
            torch.save(optimizer.state_dict(),
                       os.path.join(optimizer_dir, 'ep{:03d}_loss{:.3f}_optim.pt'.format(ep + 1, recons_loss))
                       )

        if not (ep + 1) % config['training']['val_interval']:
            val_recons_losses, total_acc_rec, chord_acc_rec, melody_acc_rec, others_acc_rec = \
                validate(ep + 1, model, val_dloader, val_dset.pad_token)
            valloss_file = os.path.join(ckpt_dir, 'valloss.txt') if start_epoch == 0 \
                else os.path.join(ckpt_dir, 'valloss_from_ep{:03d}.txt'.format(start_epoch))

            if os.path.exists(valloss_file):
                with open(valloss_file, 'a') as f:
                    f.write("ep{:03d} | loss: {:.3f} | valloss: {:.3f} (±{:.3f}) | total_acc: {:.3f} | "
                            "chord_acc: {:.3f} | melody_acc: {:.3f} | others_acc: {:.3f}\n".format(
                            ep + 1, recons_loss, np.mean(val_recons_losses), np.std(val_recons_losses),
                            np.mean(total_acc_rec), np.mean(chord_acc_rec), np.mean(melody_acc_rec), np.mean(others_acc_rec)
                    ))
            else:
                with open(valloss_file, 'w') as f:
                    f.write("ep{:03d} | loss: {:.3f} | valloss: {:.3f} (±{:.3f}) | total_acc: {:.3f} | "
                            "chord_acc: {:.3f} | melody_acc: {:.3f} | others_acc: {:.3f}\n".format(
                            ep + 1, recons_loss, np.mean(val_recons_losses), np.std(val_recons_losses),
                            np.mean(total_acc_rec), np.mean(chord_acc_rec), np.mean(melody_acc_rec), np.mean(others_acc_rec)
                    ))

        print('[epoch {:03d}] training completed\n  -- loss = {:.4f}\n  -- time elapsed = {:.2f} secs.'.format(
            ep + 1,
            recons_loss,
            ep_time,
        ))

        log_data = {
            'ep': ep + 1,
            'steps': train_steps,
            'ce_loss': recons_loss,
            'time': ep_time
        }
        log_epoch(
            os.path.join(ckpt_dir, log_file), log_data, is_init=not os.path.exists(os.path.join(ckpt_dir, log_file))
        )
