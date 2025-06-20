import os
import pickle5 as pickle
import argparse
import numpy as np

BEAT_RESOL = 480
BAR_RESOL = BEAT_RESOL * 4
TICK_RESOL = BEAT_RESOL // 24

top_five = ['Schumann', 'Chopin', 'Bach', 'Beethoven', 'Mozart']

def build_full_vocab(representation, add_velocity=False):
    vocab = []

    if representation == 'performance':
        for p in range(21, 109):
            vocab.append('Note_On_{}'.format(p))
            vocab.append('Note_Off_{}'.format(p))
        if add_velocity:
            for v in np.linspace(4, 127, 32, dtype=int):
                vocab.append('Note_Velocity_{}'.format(int(v)))
    else:
        for p in range(21, 109):
            vocab.append('Note_Pitch_{}'.format(p))
        for d in np.arange(TICK_RESOL, BAR_RESOL + TICK_RESOL, TICK_RESOL):
            vocab.append('Note_Duration_{}'.format(int(d)))

    return vocab


def events2dictionary(root, representation, event_pos=2):
    event_path = os.path.join(root, 'events')
    dictionary_path = os.path.join(root, 'dictionary.pkl')

    # list files
    event_files = os.listdir(event_path)
    n_files = len(event_files)
    print(' > num files:', n_files)

    # generate dictionary
    all_events = []
    for file in event_files:
        events = pickle.load(open(os.path.join(event_path, file), 'rb'))[0]
        for event in events:
            all_events.append('{}_{}'.format(event['name'], event['value']))
    all_events = all_events + build_full_vocab(representation=representation,
                                               add_velocity=False)

    all_events.extend([f"Composer_{each}" for each in top_five])
    unique_events = sorted(set(all_events), key=lambda x: (not isinstance(x, int), x))
    event2word = {key: i for i, key in enumerate(unique_events)}
    word2event = {i: key for i, key in enumerate(unique_events)}
    print(event2word)
    print(' > num classes:', len(word2event))
    pickle.dump((event2word, word2event), open(dictionary_path, 'wb'))


if __name__ == '__main__':
    # configuration
    parser = argparse.ArgumentParser(description='')
    required = parser.add_argument_group('required arguments')
    required.add_argument('-t', '--type',
                          choices=['score', 'performance'],
                          help='representation for music', required=True)
    args = parser.parse_args()
    representation = args.type 

    # ======== stage-one for two-stage models ========#
    events_dir = '../performance_data/full/pretrain_new'
    # events_dir = '../score_data/full/pretrain_score/'
    print(events_dir)
    events2dictionary(events_dir, representation=representation, event_pos=1)

    # events_dir = 'events/stage1/emopia_events/lead_sheet_chord11_{}'.format(representation)
    # print(events_dir)
    # events2dictionary(events_dir, add_velocity=False, add_emotion=True, add_tempo=False,
    #                   num_emotion=2, relative=relative, event_pos=1)

    # # ======== stage-two for two-stage models ========#
    # events_dir = '../dataset/embellish/classical_all_{}'.format(representation)
    # print(events_dir)
    # events2dictionary(events_dir, add_velocity=False, add_tempo=False, relative=relative, event_pos=2)

    # print(events_dir)
    # events2dictionary(events_dir, add_velocity=False, add_emotion=True, add_tempo=True,
    #                   num_emotion=4, relative=relative, event_pos=2)

    # # ======== one-stage models ========#
    # print(events_dir)
    # events2dictionary(events_dir, add_velocity=True, add_emotion=True, add_tempo=True,
    #                   num_emotion=4, relative=relative, event_pos=1)

    # events_dir = 'events/stage1/emopia_events/full_song_chord11_{}'.format(representation)
    # print(events_dir)
    # events2dictionary(events_dir, add_velocity=True, add_emotion=True, add_tempo=True,
    #                   num_emotion=4, relative=relative, event_pos=1)
