import sys
import os
import argparse
import yaml
import torch
import shutil
import numpy as np

from model.vanilla_transformer import VanillaTransformer
from events2score import convert_sequence_to_midi
from utils import pickle_load
from inference_utils import generate_events

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
    'none']


def read_vocab(vocab_file):
    event2idx, idx2event = pickle_load(vocab_file)
    orig_vocab_size = len(event2idx)
    pad_token = orig_vocab_size
    event2idx['PAD_None'] = pad_token
    vocab_size = pad_token + 1

    return event2idx, idx2event, vocab_size


def get_leadsheet_prompt(data_dir, piece, prompt_n_bars):
    bar_pos, evs = pickle_load(os.path.join(data_dir, piece))

    prompt_evs = [
        '{}_{}'.format(x['name'], x['value']) for x in evs[: bar_pos[prompt_n_bars] + 1]
    ]
    assert len(np.where(np.array(prompt_evs) == 'Bar_None')[0]) == prompt_n_bars + 1
    target_bars = len(bar_pos)

    return prompt_evs, target_bars


def event_to_txt(events, output_event_path):
    f = open(output_event_path, 'w')
    print(*events, sep='\n', file=f)


if __name__ == '__main__':
    # configuration
    parser = argparse.ArgumentParser(description='')
    required = parser.add_argument_group('required arguments')
    required.add_argument('-c', '--configuration',
                          help='configurations of training', required=True)
    required.add_argument('-r', '--representation',
                          choices=['remi', 'functional'],
                          help='representation for symbolic music', required=True)
    parser.add_argument('-o', '--output_dir',
                        help='output directory')
    parser.add_argument('-n', '--pieces',
                        default=20,
                        help='number of groups to generate')
    parser.add_argument('-p', '--composer',
                        choices=short_names,
                        help="composer's style")
    parser.add_argument('-t', '--all_composer',
                        default='True', choices=['True', 'False'],
                        help="composer's style")
    args = parser.parse_args()

    config_path = args.configuration
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    representation = args.representation
    composer = args.composer
    all_composer = args.all_composer == 'True'
    if composer is not None: all_composer = False

    if composer not in short_names and all_composer == False:
        print(f'Only support names: {short_names}')
        exit()

    inference_params = config['inference_param_path']
    out_dir = args.output_dir
    total_pieces = int(args.pieces)
    key_determine = 'rule'
    print('representation: {}, key determine: {}'.format(representation, key_determine))
    print("Load parameters from", inference_params)
    print("All Composers: {}, composer: {}".format(all_composer, composer))

    data_conf = config['data']
    data_dir = data_conf['data_dir'].format(representation)

    max_bars = 128
    temp = 1.1
    top_p = 0.99
    max_dec_len = 1200
    print('[nucleus parameters] t = {}, p = {}'.format(temp, top_p))

    torch.cuda.set_device(config['device'])

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    event2idx, idx2event, vocab_size = read_vocab(config['data']['vocab_path'].format(representation))

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

    pretrained_dict = torch.load(inference_params, map_location='cpu', weights_only=True)
    model.load_state_dict(pretrained_dict)
    model.eval()
    shutil.copy(config_path, os.path.join(out_dir, 'config_full.yaml'))

    generated_pieces = 0
    gen_times = []

    if all_composer:
        total_pieces *= len(short_names) - 1
        COMPOSERS = COMPOSERS[:-1]
    else:
        composer = COMPOSERS[short_names.index(composer)]

    while generated_pieces < total_pieces:
        if all_composer:
            composer = COMPOSERS[generated_pieces % len(COMPOSERS)]
        print("[global composer]", composer)

        gen_words, t_sec = generate_events(
                            model,
                            event2idx, idx2event,
                            max_events=max_dec_len, max_bars=max_bars,
                            primer=['Composer_{}'.format(composer), ],
                            temp=temp, top_p=top_p,
                        )

        gen_words = [idx2event[w] for w in gen_words]
        # print(gen_words[:20])
        key = gen_words[1] if representation == 'functional' else 'Key_C'
        if key is None:
            key = 'Key_C'
        # print('[global tempo]', tempo)
        out_name = 'samp_{:02d}_{}_{}'.format(generated_pieces, key.split('_')[1], composer.split()[-1])
        print(out_name)
        if os.path.exists(os.path.join(out_dir, out_name + '.mid')):
            print('[info] {} exists, skipping ...'.format(out_name))
            generated_pieces += 1
            continue

        if gen_words is None:  # model failed repeatedly
            continue
        event_to_txt(gen_words, output_event_path=os.path.join(out_dir, out_name + '.txt'))

        convert_sequence_to_midi(gen_words, is_token=True, 
                        output_midi_path=os.path.join(out_dir, out_name + '.mid'))

        gen_times.append(t_sec)
        generated_pieces += 1

    print('[info] finished generating {} pieces, avg. time: {:.2f} +/- {:.2f} secs.'.format(
        generated_pieces, np.mean(gen_times), np.std(gen_times)
    ))
