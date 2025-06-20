import torch
import pickle5 as pickle

BEAT_RESOL = 480
IDX_TO_KEY = {
    0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F',
    6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B',
}
KEY_TO_IDX = { v: k for k, v in IDX_TO_KEY.items() }

def simplify_quality(quality_str):
    qs = quality_str.strip()
    if qs in ["", "N", "None"]:
        return "None"

    # === Major chords ===
    if qs in ["maj", "M"]:
        return "M"
    if qs in ["maj7", "M7"]:
        return "M7"
    if qs in ["maj(6)"]:
        return qs.replace("(", '').replace(')', '')

    # === Minor chords ===
    if qs in ["min", "m"]:
        return "m"
    if qs in ["min7", "m7"]:
        return "m7"
    if qs == "minmaj7":
        return "minmaj7"
    if qs in ["min(6)"]:
        return qs.replace("(", '').replace(')', '')
    
    # === Augmented chords ===
    if qs in ["aug", "+"]:
        return "+"
    
    # === Diminished chords ===
    if qs in ["dim", "o"]:
        return "o"
    if qs in ["dim7", "o7"]:
        return "o7"
    if qs == 'hdim7':
        return '/o7'

    return qs

def normalize_key(note):
    if note in KEY_TO_IDX:
        return note
    flat_to_sharp = {
        "Db": "C#",
        "Eb": "D#",
        "Fb": "E",
        "Gb": "F#",
        "Ab": "G#",
        "Bb": "A#",
        "Cb": "B"
    }
    return flat_to_sharp.get(note, note)

def parse_chords(chord_file, ratio=1):
    markers = []
    current_time = 0
    
    with open(chord_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                head, bass_token, dur_token = line.rsplit(maxsplit=2)
            except ValueError:
                continue

            chord_token = head.split()[0]
            duration = float(dur_token)

            if chord_token == "N":
                chord_text = "None_None"
            else:
                if ':' in chord_token:
                    root, quality = chord_token.split(":", 1)
                else:
                    root = chord_token
                    quality = 'None'
                root = normalize_key(root)
                if '/' in quality:
                    base_quality = quality.split('/')[0]
                else:
                    base_quality = quality
                simple_quality = simplify_quality(base_quality)

                chord_text = f"{root}_{simple_quality}"

            markers.append((current_time, chord_text))
            current_time += int(round(duration * ratio))

        return markers
    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def numpy_to_tensor(arr, use_gpu=True):
    if use_gpu:
        return torch.tensor(arr).to(device).float()
    else:
        return torch.tensor(arr).float()

def tensor_to_numpy(tensor):
    return tensor.cpu().detach().numpy()

def pickle_load(f):
    return pickle.load(open(f, 'rb'))

def pickle_dump(obj, f):
    pickle.dump(obj, open(f, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)