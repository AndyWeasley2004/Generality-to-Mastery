import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
import pickle
from model.vanilla_encoder import ComposerClassifier
from dataloader import MusicSegmentDataset
import matplotlib.pyplot as plt
import argparse
import yaml

def pickle_load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def convert_event(event_tokens, event2idx, to_ndarr=True):
    token_ids = [event2idx.get(tok, 0) for tok in event_tokens]
    if to_ndarr:
        return np.array(token_ids, dtype=np.int16)
    return token_ids

def is_nested_folder(data_folder):
    for item in os.listdir(data_folder):
        if os.path.isdir(os.path.join(data_folder, item)):
            return True
    return False

class NestedComposerDataset(Dataset):
    """
    Each sub-folder corresponds to a composer label.
    We assume that each file is a .pkl file.
    We use similar loading/processing as MusicSegmentDataset,
    but override the composer field with the subfolder name.
    """
    def __init__(self, root_folder, vocab_path, model_dec_seqlen=1200, data_type='score'):
        self.root_folder = root_folder
        self.vocab_path = vocab_path
        self.model_dec_seqlen = model_dec_seqlen
        self.data_type = data_type
        self.composers = sorted([d for d in os.listdir(root_folder)
                                 if os.path.isdir(os.path.join(root_folder, d))])
        self.files = []
        self.labels = []
        for composer in self.composers:
            comp_folder = os.path.join(root_folder, composer)
            # Assume pkl files are used.
            pkl_files = [os.path.join(comp_folder, f) for f in os.listdir(comp_folder) if f.endswith('.pkl')]
            for pf in pkl_files:
                self.files.append(pf)
                self.labels.append(composer.lower())
                
        # Load vocabulary (assume pickle load function available)
        event2idx, idx2event = pickle_load(self.vocab_path)
        self.event2idx = event2idx
        self.idx2event = idx2event
        # In eval mode for score data, we want not to include composer token in input.
        self.eval_mode = True

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load file (assume pickle load function)
        import pickle
        with open(self.files[idx], 'rb') as f:
            actual_events, seg_indices = pickle.load(f)
        # Ensure seg_indices starts with 0.
        if len(seg_indices) == 0 or seg_indices[0] != 0:
            seg_indices.insert(0, 0)
        # Build events string.
        events_str = ['{}_{}'.format(x['name'], x['value']) for x in actual_events]
        # Composer token is overridden by the folder label.
        true_composer = self.labels[idx]
        # Get tempo token.
        tempo_token = events_str[1]
        start = 0
        end = min(len(events_str), self.model_dec_seqlen)
        # In eval mode for score, we omit the composer token.
        seg_tokens = [tempo_token] + events_str[start+2:end]
        token_ids = convert_event(seg_tokens, self.event2idx, to_ndarr=False)
        dec_inp = token_ids[:-1]
        dec_tgt = token_ids[1:]
        sample = {
            'dec_inp': np.array(dec_inp, dtype=np.int16),
            'dec_tgt': np.array(dec_tgt, dtype=np.int16),
            'length': len(dec_inp),
            'filename': os.path.basename(self.files[idx]),
            'id': idx,
            'composer': true_composer  # override with folder name
        }
        return sample

    def collate_fn(self, batch):
        # Same collate_fn as in MusicSegmentDataset, with composer list.
        dec_inp_list = []
        dec_tgt_list = []
        length_list = []
        id_list = []
        filename_list = []
        composer_list = []  # collect composer tokens

        # Use self.model_dec_seqlen as cap.
        for seg in batch:
            curr_len = seg['length']
            trunc_len = min(curr_len, self.model_dec_seqlen)
            tmp_inp = seg['dec_inp'][:trunc_len]
            tmp_inp = np.pad(tmp_inp, (0, self.model_dec_seqlen - trunc_len),
                             'constant', constant_values=0)  # assume pad token 0
            tmp_tgt = seg['dec_tgt'][:trunc_len]
            tmp_tgt = np.pad(tmp_tgt, (0, self.model_dec_seqlen - trunc_len),
                             'constant', constant_values=0)

            dec_inp_list.append(tmp_inp)
            dec_tgt_list.append(tmp_tgt)
            length_list.append(curr_len)
            filename_list.append(seg['filename'])
            id_list.append(seg['id'])
            composer_list.append(seg['composer'])
        batch_dict = {
            'id': torch.LongTensor(np.stack(id_list, axis=0)),
            'dec_inp': torch.LongTensor(np.stack(dec_inp_list, axis=0)),
            'dec_tgt': torch.LongTensor(np.stack(dec_tgt_list, axis=0)), 
            'length': torch.LongTensor(np.stack(length_list, axis=0)),
            'filename': filename_list,
            'composer': composer_list
        }
        return batch_dict

class FlatInferenceDataset(Dataset):
    """
    A customized dataset for inference when the data folder is flat (i.e., contains .pkl files directly).
    This dataset runs in eval mode (i.e., it does not prepend a composer token), uses only the tempo token
    (or similar) for the input, and caps the sequence length to a specified value (default 1200).
    """
    def __init__(self, data_dir, vocab_path, model_dec_seqlen=1200, data_type='score'):
        self.data_dir = data_dir
        self.vocab_path = vocab_path
        self.model_dec_seqlen = model_dec_seqlen
        self.data_type = data_type
        self.file_list = sorted(glob(os.path.join(self.data_dir, '*.pkl')))
        
        # Load vocabulary from pickle.
        vocab_pack = pickle_load(self.vocab_path)
        self.event2idx = vocab_pack[0]
        self.idx2event = vocab_pack[1]
        # Set pad token as in original dataset.
        self.pad_token = len(self.event2idx)
        self.event2idx['PAD_None'] = self.pad_token
        self.vocab_size = self.pad_token + 1

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        actual_events, seg_indices = pickle_load(file_path)
        
        # Ensure seg_indices starts with 0.
        if len(seg_indices) == 0 or seg_indices[0] != 0:
            seg_indices.insert(0, 0)
            
        # Create events string (same as in your original __getitem__).
        events_str = ['{}_{}'.format(x['name'], x['value']) for x in actual_events]
        
        # In eval mode for score data, we omit the composer token.
        # Typically, the first two tokens might be [composer_token, tempo_token]; we only keep tempo token.
        # Here we simply remove the composer token if present.
        if self.data_type == 'score':
            seg_tokens = events_str[1:]
        else:
            seg_tokens = events_str
            
        # Cap the sequence length.
        seg_tokens = seg_tokens[:self.model_dec_seqlen]
        token_ids = convert_event(seg_tokens, self.event2idx, to_ndarr=False)
        dec_inp = token_ids[:-1]
        dec_tgt = token_ids[1:]
        
        sample = {
            'dec_inp': np.array(dec_inp, dtype=np.int16),
            'dec_tgt': np.array(dec_tgt, dtype=np.int16),
            'length': len(dec_inp),
            'filename': os.path.basename(file_path),
            'composer': None
        }
        return sample

    def collate_fn(self, batch):
        dec_inp_list = []
        dec_tgt_list = []
        length_list = []
        id_list = []
        filename_list = []
        composer_list = []  # will be all None
        
        for i, seg in enumerate(batch):
            curr_len = seg['length']
            trunc_len = min(curr_len, self.model_dec_seqlen)
            tmp_inp = seg['dec_inp'][:trunc_len]
            tmp_inp = np.pad(tmp_inp, (0, self.model_dec_seqlen - trunc_len),
                             'constant', constant_values=self.pad_token)
            
            tmp_tgt = seg['dec_tgt'][:trunc_len]
            tmp_tgt = np.pad(tmp_tgt, (0, self.model_dec_seqlen - trunc_len),
                             'constant', constant_values=self.pad_token)
            
            dec_inp_list.append(tmp_inp)
            dec_tgt_list.append(tmp_tgt)
            length_list.append(curr_len)
            filename_list.append(seg['filename'])
            id_list.append(i)
            composer_list.append(seg['composer'])
        
        batch_dict = {
            'id': torch.LongTensor(np.stack(id_list, axis=0)),
            'dec_inp': torch.LongTensor(np.stack(dec_inp_list, axis=0)),
            'dec_tgt': torch.LongTensor(np.stack(dec_tgt_list, axis=0)), 
            'length': torch.LongTensor(np.stack(length_list, axis=0)),
            'filename': filename_list,
            'composer': composer_list
        }
        return batch_dict

# Evaluation function
def evaluate(model, dataloader, composer2idx, remove_composer_token=True, 
             device='cuda', nested=False, data_folder=None):
    """
    For flat data: count predicted composer frequencies.
    For nested data: compute confusion matrix.
    """
    model.eval()
    total_samples = 0
    predictions = []
    true_labels = []
    with torch.no_grad():
        for batch in dataloader:
            # For eval mode, remove composer token from input (assume it's the first token if present)
            if remove_composer_token:
                inp = batch['dec_inp'][:, 1:]
            else:
                inp = batch['dec_inp']
            inp = inp.to(device)
            logits = model(inp)
            preds = logits.argmax(dim=-1).cpu().numpy()
            predictions.extend(preds)
            if nested:
                # In nested mode, we have true labels.
                # Convert composer string label into index.
                for comp in batch['composer']:
                    true_labels.append(composer2idx[comp.lower()])
            total_samples += inp.size(0)

    if nested:
        # Compute confusion matrix with text labels.
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(true_labels, predictions, labels=list(composer2idx.values()))
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_normalized = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

        plt.figure(figsize=(8,6), dpi=200)
        plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
        # plt.title("From Scratch Samples Classification Results", fontsize=18)
        plt.colorbar()
        tick_marks = np.arange(len(composer2idx))
        composer_names = [name.capitalize() for name, _ in sorted(composer2idx.items(), key=lambda x: x[1])]
        plt.xticks(tick_marks, composer_names, fontsize=14, rotation=45)
        plt.yticks(tick_marks, composer_names, fontsize=14)
        # plt.ylabel('Conditioned Composer', fontsize=16)
        # plt.xlabel('Predicted Composer', fontsize=16)

        # Annotate each cell with percentage relative to its row total.
        for i in range(cm.shape[0]):
            row_sum = np.sum(cm[i, :])
            for j in range(cm.shape[1]):
                if row_sum > 0:
                    ratio = cm[i, j] / row_sum
                else:
                    ratio = 0
                plt.text(j, i, f"{ratio:.2f}", ha="center", va="center",
                        color="white" if ratio > 0.5 else "black")

        plt.tight_layout()
        cm_path = os.path.join(data_folder, "confusion_matrix.png")
        plt.savefig(cm_path)
        print(f"Confusion matrix saved to {cm_path}")
        return cm
    else:
        # In flat mode, count frequency of predicted labels and plot a bar chart.
        from collections import Counter
        pred_counter = Counter(predictions)
        inv_composer = {v: k for k, v in composer2idx.items()}
        pred_counts = {inv_composer[idx]: count for idx, count in pred_counter.items()}
        print("Predicted composer counts:", pred_counts)

        # Ensure all composers are shown in the bar plot, even if count is zero.
        composer_names = sorted(list(composer2idx.keys()))
        counts = [pred_counts.get(name, 0) for name in composer_names]

        # Compute ratio of each composer relative to total samples.
        total_samples = sum(counts)
        ratios = [count / total_samples if total_samples > 0 else 0 for count in counts]

        plt.figure(figsize=(8,6), dpi=150)
        plt.bar(composer_names, ratios, color=['#29C5FD', '#0AD016', '#F16D35', '#E053EA'])
        plt.xlabel("Composer")
        plt.ylabel("Ratio")
        plt.ylim(0, 1)
        plt.title("REMI Predicted Composer Distribution")
        plt.tight_layout()
        barplot_path = os.path.join(data_folder, "composer_distribution_ratio.png")
        plt.savefig(barplot_path)
        print(f"Bar plot saved to {barplot_path}")
        return pred_counts

# Main evaluation pipeline
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="External Data Evaluation for Composer Classifier")
    parser.add_argument('-c', '--configuration', help='Config file', required=True)
    parser.add_argument('-d', '--data_folder', help='Root folder for external data', required=True)
    args = parser.parse_args()

    config = yaml.load(open(args.configuration, 'r'), Loader=yaml.FullLoader)
    device = config['device']
    torch.cuda.set_device(device)

    event2idx, idx2event = pickle.load(open(config['data']['vocab_path'], 'rb'))
    used_categories = ['bach', 'chopin', 'mozart', 'beethoven']
    composer2idx = {cat.lower(): idx for idx, cat in enumerate(used_categories)}

    mconf = config['model']['encoder']
    model = ComposerClassifier(
        d_word_embed=mconf['d_model'],
        event2word=event2idx,
        n_layer=mconf['n_layer'],
        n_head=mconf['n_head'],
        d_model=mconf['d_model'],
        d_ff=mconf['d_ff'],
        max_seq_len=mconf['tgt_len'],
        num_classes=len(composer2idx),
        dropout=mconf['dropout'],
        pre_lnorm=config['model']['pre_lnorm']
    ).to(device)
    model.load_state_dict(torch.load(config['inference_param_path'], map_location=device), strict=False)
    print(f"Loaded checkpoint from {config['inference_param_path']}")

    nested = is_nested_folder(args.data_folder)
    if nested:
        eval_dataset = NestedComposerDataset(
            root_folder=args.data_folder,
            vocab_path=config['data']['vocab_path'],
            model_dec_seqlen=1200,
            data_type='score'
        )
        print(f"Loaded nested dataset with {len(eval_dataset)} files.")
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=config['data']['batch_size'],
            num_workers=4,
            collate_fn=eval_dataset.collate_fn
        )
    else:
        eval_dataset = FlatInferenceDataset(
            vocab_path=config['data']['vocab_path'],
            data_dir=args.data_folder,
            model_dec_seqlen=1200,
        )
        print(f"Loaded flat dataset with {len(eval_dataset)} files.")
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=config['data']['batch_size'],
            num_workers=4,
            collate_fn=eval_dataset.collate_fn
        )

    results = evaluate(model, eval_dataloader, composer2idx, remove_composer_token=True,
                       device=device, nested=nested, data_folder=args.data_folder)
