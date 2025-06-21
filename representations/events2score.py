from miditoolkit import MidiFile, Instrument, Note, TempoChange, TimeSignature
import os
from tqdm import tqdm
import pickle5 as pickle

def pickle_load(f):
    return pickle.load(open(f, 'rb'))

def get_bar_parameters(ts_token_value, ticks_per_beat=480):
    num_str, denom_str = ts_token_value.split('/')
    numerator = int(num_str)
    denominator = int(denom_str)
    bar_length = int(numerator * ticks_per_beat * (4 / denominator))

    if numerator == 6 and denominator == 8:
        effective_beats = 6
        grids_per_beat = 6
    elif numerator == 3 and denominator == 8:
        effective_beats = 3
        grids_per_beat = 6
    else:
        effective_beats = numerator
        grids_per_beat = 12

    grid_count = effective_beats * grids_per_beat
    ticks_per_grid = bar_length // grid_count
    return bar_length, grid_count, ticks_per_grid


def token_to_event(tokens):
    events = []
    for token in tokens:
        parts = token.split('_', 2)
        if len(parts) == 3 and parts[0] in ["Note", "Time", "Chord"]:
            # Tokens like "Note_Pitch", "Note_Duration", "Time_Signature" or "Chord_C_m"
            name = parts[0] + "_" + parts[1]   # e.g. "Note_Pitch"
            value = parts[2]                  # e.g. "60"
        elif len(parts) == 2:
            # like composers and beat
            name, value = parts
            if name == 'Beat': name = int(name)
        else:
            name, value = parts[0], "_".join(parts[1:])
        events.append({'name': name, 'value': value})
    return events

def convert_sequence_to_midi(sequence, output_midi_path, is_token=False):
    if is_token:
        sequence = token_to_event(sequence)
        
    midi = MidiFile()
    midi.ticks_per_beat = 480

    time_signatures = []
    tempo_changes = []

    instrument = Instrument(program=0, is_drum=False, name="Piano")
    midi.instruments.append(instrument)

    current_ts_value = "4/4"
    bar_length, grid_count, ticks_per_grid = get_bar_parameters(current_ts_value, midi.ticks_per_beat)
    current_tempo_bpm = 120
    inserted_any_tempo = False

    absolute_tick = 0
    notes = []
    current_grid = 0

    i = 0
    while i < len(sequence):
        token = sequence[i]
        name = token["name"]
        value = token["value"]

        # --- Global Tempo Token (should appear near the beginning) ---
        if name == "Tempo":
            current_tempo_bpm = value
            tempo_changes.append(TempoChange(current_tempo_bpm, absolute_tick))
            inserted_any_tempo = True
            i += 1

        # --- Skip other global tokens ---
        elif name in ["Composer", "BOS"]:
            i += 1

        # --- Process a Bar block ---
        elif name == "Bar":
            bar_start_tick = absolute_tick
            i += 1  # Skip the "Bar" token

            # Immediately after a Bar, we expect a Time_Signature token.
            if i < len(sequence) and sequence[i]["name"] == "Time_Signature":
                ts_token_value = sequence[i]["value"]
                current_ts_value = ts_token_value
                num_str, denom_str = current_ts_value.split('/')
                time_signatures.append(TimeSignature(int(num_str), int(denom_str), bar_start_tick))
                bar_length, grid_count, ticks_per_grid = get_bar_parameters(current_ts_value, midi.ticks_per_beat)
                i += 1
            else:
                pass

            # Process events inside the bar until the next "Bar" or "EOS" token.
            while i < len(sequence) and sequence[i]["name"] not in ["Bar", "EOS"]:
                sub_token = sequence[i]
                sub_name = sub_token["name"]
                sub_value = sub_token["value"]

                if sub_name == "Beat":
                    current_grid = sub_value
                    i += 1

                elif sub_name == "Note_Pitch":
                    pitch = sub_value
                    i += 1
                    if i < len(sequence) and sequence[i]["name"] == "Note_Duration":
                        duration = sequence[i]["value"]
                        i += 1
                    else:
                        duration = ticks_per_grid  # fallback if missing
                    note_start = bar_start_tick + (current_grid * ticks_per_grid)
                    note_end = note_start + duration
                    notes.append({"pitch": pitch, "start": note_start, "end": note_end})

                else:
                    i += 1

            absolute_tick += bar_length

        # --- End of Sequence token ---
        elif name == "EOS":
            break

        else:
            i += 1

    if not inserted_any_tempo:
        tempo_changes.append(TempoChange(current_tempo_bpm, 0))

    if not time_signatures or time_signatures[0].time != 0:
        num_str, denom_str = current_ts_value.split('/')
        time_signatures.insert(0, TimeSignature(int(num_str), int(denom_str), 0))

    notes.sort(key=lambda x: x["start"])
    for n in notes:
        instrument.notes.append(
            Note(
                velocity=64,  # default velocity
                pitch=n["pitch"],
                start=n["start"],
                end=n["end"]
            )
        )

    # Sort and assign time signature and tempo changes.
    time_signatures.sort(key=lambda ts: ts.time)
    midi.time_signature_changes = time_signatures

    tempo_changes.sort(key=lambda tc: tc.time)
    midi.tempo_changes = tempo_changes
    midi.dump(output_midi_path)


if __name__ == '__main__':
    dataset = 'finetune_top5_selected_new'
    data_root = f'../score_data/full/{dataset}/'
    output_root = os.path.join(data_root, '../', dataset + '_reconstr')
    os.makedirs(output_root, exist_ok=True)

    # Gather all file paths recursively.
    all_files = []
    for root, dirs, files in os.walk(data_root):
        for file in files:
            # Skip non-data files if necessary.
            all_files.append(os.path.join(root, file))

    for filepath in tqdm(all_files):
        try:
            # Compute the relative path from the data_root.
            relative_path = os.path.relpath(filepath, data_root)
            # Change the extension from (e.g.) .pkl to .mid and build the output path.
            output_file_path = os.path.join(output_root, os.path.splitext(relative_path)[0] + '.mid')
            # Create any missing subdirectories in the output folder.
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

            data = pickle_load(filepath)[0]
            convert_sequence_to_midi(data, output_midi_path=output_file_path)
        except:
            continue