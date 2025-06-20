import sys
import os
import random
import argparse
import yaml
import torch
import numpy as np
import time

from model.vanilla_transformer import VanillaTransformer
from utils import pickle_load
from dataloader import MusicSegmentDataset

sys.path.append('./model/')
sys.path.append('./')

COMPOSERS = [
    # 'Schumann', 
    'Chopin', 
    'Bach',
    'Beethoven', 
    'Mozart',
    'None'
]
short_names = [
    # 'schumann',
    'chopin',
    'bach',
    'beethoven',
    'mozart',
    'none'
]

def read_vocab(vocab_file):
    event2idx, idx2event = pickle_load(vocab_file)
    orig_vocab_size = len(event2idx)
    pad_token = orig_vocab_size
    event2idx['PAD_None'] = pad_token
    vocab_size = pad_token + 1
    return event2idx, idx2event, vocab_size

def get_leadsheet_prompt(data_dir, piece, prompt_n_bars):
    """
    Given a file path (piece) from the validation set, load its contents and use the first
    `prompt_n_bars` to form a primer.
    """
    # piece is the full file path.
    evs, bar_pos = pickle_load(piece)
    prompt_evs = [
        '{}_{}'.format(x['name'], x['value']) for x in evs[: bar_pos[prompt_n_bars] + 1]
    ]
    # Ensure the primer contains exactly prompt_n_bars+1 Bar tokens.
    assert len(np.where(np.array(prompt_evs) == 'Bar_None')[0]) == prompt_n_bars + 1
    return prompt_evs, prompt_n_bars

########################################
# Sampling utilities
########################################
def temperature(logits, temperature):
    try:
        probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
        assert np.count_nonzero(np.isnan(probs)) == 0
    except Exception as e:
        print('overflow detected, using higher precision', e)
        logits = logits.astype(np.float128)
        from scipy.special import softmax
        probs = softmax(logits / temperature).astype(float)
        assert np.count_nonzero(np.isnan(probs)) == 0
    return probs

def nucleus(probs, p):
    probs = probs / np.sum(probs)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    cusum_sorted_probs = np.cumsum(sorted_probs)
    after_threshold = cusum_sorted_probs > p
    if np.sum(after_threshold) > 0:
        indices = np.where(after_threshold)[0]
        if len(indices) > 1:
            last_index = indices[1]
        else:
            last_index = indices[0] + 1
        candi_index = sorted_index[:last_index]
    else:
        candi_index = sorted_index[:3]  # fallback candidate set
    candidate_count = len(candi_index)
    candi_probs = np.array([probs[i] for i in candi_index], dtype=np.float64)
    candi_probs /= np.sum(candi_probs)
    word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return word, candidate_count

def get_position_idx(event):
    return int(event.split('_')[-1])

def generate_events(model, event2idx, idx2event, max_events=2048, primer=None, temp=1.2, top_p=0.9,
                    prompt_bars=None, additional_bars=1):
    if primer is None:
        raise ValueError("Primer must be provided for primer sampling.")
    generated = [event2idx[e] for e in primer]
    generated_bars = prompt_bars if prompt_bars is not None else 0
    target_bars = generated_bars + additional_bars

    candidate_counts = []  # candidate counts for each autoregressive step (in the additional bar)
    device = next(model.parameters()).device
    steps = 0
    time_st = time.time()
    cur_pos = 0
    failed_cnt = 0

    while generated_bars < target_bars:
        dec_input = torch.LongTensor(generated).unsqueeze(1).to(device)
        logits = model.generate(dec_input)
        logits = logits.detach().cpu().numpy()
        probs = temperature(logits, temperature=temp)
        word, cand_count = nucleus(probs, p=top_p)
        candidate_counts.append(cand_count)
        word_event = idx2event[word]

        # Check for Beat token to ensure positions are increasing.
        if 'Beat' in word_event:
            event_pos = get_position_idx(word_event)
            if event_pos < cur_pos:
                failed_cnt += 1
                print('[info] position not increasing, failed cnt:', failed_cnt)
                if failed_cnt >= 256:
                    print('[FATAL] model stuck, exiting ...')
                    break
                continue
            else:
                cur_pos = event_pos
                failed_cnt = 0

        # Once a Bar token is produced, assume the additional bar is complete.
        if 'Bar' in word_event:
            generated_bars += 1
            cur_pos = 0
            print('[info] generated additional bar, total bars = {}'.format(generated_bars))
            generated.append(word)
            break

        if word_event == 'PAD_None':
            continue

        generated.append(word)
        if len(generated) == 2:
            generated.append(event2idx['BOS_None'])
        steps += 1

        if len(generated) > max_events:
            print('[info] max events reached')
            break
        if word_event == 'EOS_None':
            print('[info] gotten EOS')
            break

    elapsed = time.time() - time_st
    # print('-- finished generation loop in {:.2f} secs, steps = {}'.format(elapsed, steps))
    candidate_counts = np.array(candidate_counts)
    stats = {
        'mean': float(np.mean(candidate_counts)),
        # '75%': float(np.percentile(candidate_counts, q=75)),
        # '25%': float(np.percentile(candidate_counts, q=25)),
        # '50%': float(np.percentile(candidate_counts, q=50)),
        'std': float(np.std(candidate_counts)),
        'min': int(np.min(candidate_counts)),
        'max': int(np.max(candidate_counts)),
        'steps': int(len(candidate_counts))
    }
    # print('[stats] Candidate options: mean={:.2f}, median={:.2f}, std={:.2f}, min={}, max={}, steps={}'.format(
    #     stats['mean'], stats['median'], stats['std'], stats['min'], stats['max'], stats['steps']
    # ))
    return elapsed, stats

########################################
# Main analysis pipeline
########################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Candidate statistics analysis using validation set')
    required = parser.add_argument_group('required arguments')
    required.add_argument('-c', '--configuration', help='configuration file path', required=True)
    parser.add_argument('-o', '--output_dir', help='output directory', required=True)
    parser.add_argument('-n', '--pieces', default=20, help='number of pieces to analyze')
    parser.add_argument('-p', '--composer', choices=short_names, help="composer's style")
    parser.add_argument('-t', '--all_composer', default='True', choices=['True', 'False'], help="whether to iterate over all composers")
    args = parser.parse_args()

    config_path = args.configuration
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    composer = args.composer
    all_composer = args.all_composer == 'True'
    if composer is not None:
        all_composer = False

    if composer not in short_names and not all_composer:
        print(f'Only support names: {short_names}')
        exit()

    inference_params = config['inference_param_path']
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    total_pieces = int(args.pieces)
    key_determine = 'rule'
    print("Load parameters from", inference_params)
    print("All Composers: {}, composer: {}".format(all_composer, composer))

    temp = 1.1
    top_p = 0.99
    max_dec_len = 1200
    print('[nucleus parameters] t = {}, p = {}'.format(temp, top_p))

    torch.cuda.set_device(config['device'])

    # Create the validation dataset using your MusicSegmentDataset logic.
    val_dset = MusicSegmentDataset(
        data_dir=config['data']['data_dir'],
        vocab_path=config['data']['vocab_path'],
        model_dec_seqlen=config['model']['decoder']['tgt_len'],
        augment=False,
        from_start_only=True,
        split='valid',
        train_ratio=0.97,
        used_categories=['bach', 'chopin', 'mozart', 'beethoven']
        # used_categories=['pop', 'folk', 'classical']
    )

    # Collect all validation file paths from the dataset.
    val_files = []
    for category, files in val_dset.category_files.items():
        val_files.extend(files)
    print('[info] Total validation pieces available: {}'.format(len(val_files)))

    # Sample a subset of validation pieces for analysis.
    if len(val_files) > total_pieces:
        prompt_pieces = random.sample(val_files, total_pieces)
    else:
        prompt_pieces = val_files

    # Save the list of sampled validation pieces for record keeping.
    with open(os.path.join(out_dir, 'sampled_validation_pieces.txt'), 'w') as f:
        f.write("\n".join(prompt_pieces))

    # Initialize model and vocabulary.
    mconf = config['model']
    event2idx, idx2event, _ = read_vocab(config['data']['vocab_path'])
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

    pretrained_dict = torch.load(inference_params, map_location='cpu', weights_only=True)
    model.load_state_dict(pretrained_dict)
    model.eval()

    candidate_stats_list = []
    gen_times = []

    for i, piece_file in enumerate(prompt_pieces):
        print('[info] Processing piece {}: {}'.format(i, piece_file))
        try:
            # Use 4 bars as primer.
            primer_tokens, primer_bar_count = get_leadsheet_prompt(config['data']['data_dir'], piece_file, 
                                                                   prompt_n_bars=4)
        except Exception as e:
            print('[warning] Failed to get primer for piece {}: {}'.format(piece_file, e))
            continue

        t_sec, stats = generate_events(
            model,
            event2idx, idx2event,
            max_events=max_dec_len,
            primer=primer_tokens,
            temp=temp, top_p=top_p,
            prompt_bars=primer_bar_count,
            additional_bars=1,
        )
        gen_times.append(t_sec)
        candidate_stats_list.append(stats)

    # Compute overall candidate statistics arrays from candidate_stats_list
    candidate_means = np.array([s['mean'] for s in candidate_stats_list])
    candidate_steps = np.array([s['steps'] for s in candidate_stats_list])

    overall_mean = np.mean(candidate_means)
    overall_steps = np.mean(candidate_steps)

    # Number of validation pieces (samples)
    n = len(candidate_stats_list)

    # 95% confidence intervals using Z=1.96 (assuming n is large enough)
    mean_std_error = np.std(candidate_means, ddof=1) / np.sqrt(n)
    mean_ci = 1.96 * mean_std_error
    lower_percentile = np.percentile(candidate_means, q=25)
    half_percentile = np.percentile(candidate_means, q=50)
    upper_percentile = np.percentile(candidate_means, q=75)
    min = np.percentile(candidate_means, q=10)
    max = np.percentile(candidate_means, q=90)

    # Update overall_stats to include confidence intervals.
    overall_stats = {
        'overall_mean': overall_mean,
        'overall_mean_95ci': f'{overall_mean} ± {mean_ci}',
        '25%_percentile': lower_percentile,
        '50%_percentile': half_percentile,
        '75%_percentile': upper_percentile,
        'min': min,
        'max': max,
        'num_pieces': n
    }

    stats_file = os.path.join(out_dir, 'candidate_stats.txt')
    with open(stats_file, 'w') as f:
        f.write("Candidate Statistics for First Additional Bar Generation\n")
        f.write("=======================================================\n")
        # for idx, s in enumerate(candidate_stats_list):
        #     f.write("Piece {}: mean={:.2f}, median={:.2f}, std={:.2f}, min={}, max={}, steps={}\n".format(
        #         idx, s['mean'], s['median'], s['std'], s['min'], s['max'], s['steps']
        #     ))
        f.write("Overall Statistics:\n")
        f.write("Mean: {:.2f} (95% CI: {})\n".format(
            overall_stats['overall_mean'], overall_stats['overall_mean_95ci']
        ))
        f.write("25% Percentile: {:.2f}\n".format(overall_stats['25%_percentile']))
        f.write("50% Percentile: {:.2f}\n".format(overall_stats['50%_percentile']))
        f.write("75% Percentile: {:.2f}\n".format(overall_stats['75%_percentile']))
        f.write("minimum: {:.2f}\n".format(overall_stats['min']))
        f.write("maximum: {:.2f}\n".format(overall_stats['max']))



