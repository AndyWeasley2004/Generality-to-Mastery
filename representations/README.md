# Data Processing

## Convert MIDI to events

### Dataset
1. Download and unzip [pretrain_score]([https://zenodo.org/records/13122742](https://huggingface.co/Itsuki-music/Generality_to_Mastery/blob/main/full/pretrain_score.zip)) and [finetune score from target composers](https://huggingface.co/Itsuki-music/Generality_to_Mastery/blob/main/full/finetune_top4_selected_new.zip); you may rename the directory
2. Update the `data_root`, `event_root` and `add_composer` in the `midi2events_score.py` with your midi data path, desired output event path, and whether to add actual composer token instead of putting `Composer_None` (`add_composer=True` for fine-tuning dataset and `False` for pre-training dataset)
3. The pipeline handles the multi-genres file structure already for one corpus with multiple-style data. You can run the script as follows
```angular2html
python midi2events_score.py
```

## Build Vocabulary
We provide a dummy option for performnace representation using `note_on`, `note_off` and `time_shift` tokens, but our work focuses on symbolic music, so please build vocabulary as follows.
```angular2html
python events2words.py --type score
```

## Data splits
Please refer to details in [training script](https://github.com/AndyWeasley2004/Generality-to-Mastery/blob/main/train_score.py) and [dataset interface](https://github.com/AndyWeasley2004/Generality-to-Mastery/blob/main/dataloader.py) for the train-validation split. We employ a stratified split with a fixed validation ratio using a random seed, which ensures both reproducibility and randomness.

