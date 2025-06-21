import torch
import pickle
import json
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def numpy_to_tensor(arr, use_gpu=True):
    if use_gpu:
        return torch.tensor(arr).to(device).float()
    else:
        return torch.tensor(arr).float()

def tensor_to_numpy(tensor):
    return tensor.cpu().detach().numpy()


def list2str(a_list):
    return ''.join([str(i) for i in a_list])


def pickle_load(f):
    return pickle.load(open(f, 'rb'))


def pickle_dump(obj, f):
    pickle.dump(obj, open(f, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


def json_read(path):
    with open(path, 'r') as f:
        content = json.load(f)
    f.close()
    return content


def compute_accuracy(dec_logits, dec_target, inp_chord, inp_melody, pad_token):
    dec_pred = torch.argmax(dec_logits, dim=-1).permute(1, 0).cpu().numpy()
    dec_target = dec_target.permute(1, 0).cpu().numpy()
    
    valid = dec_target != pad_token
    total_acc = np.mean((dec_pred[valid] == dec_target[valid]).astype(np.float16))

    chord_idx = (inp_chord.cpu().numpy() == 1)
    if np.sum(chord_idx) > 0:
        chord_acc = np.mean((dec_pred[chord_idx] == dec_target[chord_idx]).astype(np.float16))
    else:
        chord_acc = 0.0

    melody_idx = (inp_melody.cpu().numpy() == 1)
    melody_acc = np.mean((dec_pred[melody_idx] == dec_target[melody_idx]).astype(np.float16))

    total_count = np.sum(valid)
    chord_count = np.sum(chord_idx)
    melody_count = np.sum(melody_idx)
    others_count = total_count - chord_count - melody_count
    if others_count > 0:
        others_acc = (total_acc * total_count - chord_acc * chord_count - melody_acc * melody_count) / others_count
    else:
        others_acc = 0.0

    return total_acc, chord_acc, melody_acc, others_acc
