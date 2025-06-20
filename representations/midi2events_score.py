import mido
import os
import pickle
from tqdm import tqdm

def create_event(name, value):
    return {'name': name, 'value': value}

def align_tempo(avg_tempo, allowed_tempos=[40, 80, 120, 160]):
    return min(allowed_tempos, key=lambda t: abs(t - avg_tempo))

def midi_to_sequence(midi_file_path, composer):
    """
    Convert a MIDI file (score-like) into a sequence representation of events
    suitable for training a transformer. Returns the sequence (list of events)
    and a list of indices where Bar tokens occur.
    """
    # Load MIDI file and check ticks per beat (TPB)
    mid = mido.MidiFile(midi_file_path)
    original_tpq = mid.ticks_per_beat
    # Conversion factor to bring ticks per beat to 480 (if needed)
    factor = 480 / original_tpq if original_tpq != 480 else 1.0

    # Collect all messages with absolute tick times (applying conversion)
    all_messages = []
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            all_messages.append((abs_tick * factor, msg))
    all_messages.sort(key=lambda x: x[0])

    # --- Compute Global Tempo (Time-weighted average) ---
    default_tempo = 500000  # Default tempo: 500000 microseconds per beat (~120 BPM)
    current_tempo = default_tempo
    last_tick = 0
    tempo_time_sum = 0.0  # sum(duration * BPM)
    total_duration = 0.0
    for abs_tick, msg in all_messages:
        if msg.type == 'set_tempo':
            seg_duration = abs_tick - last_tick
            seg_bpm = 60000000 / current_tempo  
            tempo_time_sum += seg_bpm * seg_duration
            total_duration += seg_duration
            current_tempo = msg.tempo
            last_tick = abs_tick
    max_tick = all_messages[-1][0] if all_messages else 0
    if max_tick > last_tick:
        seg_duration = max_tick - last_tick
        seg_bpm = 60000000 / current_tempo
        tempo_time_sum += seg_bpm * seg_duration
        total_duration += seg_duration
    avg_tempo = tempo_time_sum / total_duration if total_duration > 0 else 120
    global_tempo = align_tempo(avg_tempo)

    # --- Extract Note Events ---
    note_events = []
    ongoing_notes = {}
    for abs_tick, msg in all_messages:
        if msg.type == 'note_on' and msg.velocity > 0:
            ongoing_notes[(msg.channel, msg.note)] = abs_tick
        elif (msg.type == 'note_off') or (msg.type == 'note_on' and msg.velocity == 0):
            key = (msg.channel, msg.note)
            if key in ongoing_notes:
                start_tick = ongoing_notes.pop(key)
                duration = abs_tick - start_tick
                note_events.append({
                    'start': start_tick,
                    'duration': duration,
                    'pitch': msg.note
                })
    note_events.sort(key=lambda x: x['start'])

    # --- Extract Time Signature Events ---
    time_sigs = []
    for abs_tick, msg in all_messages:
        if msg.type == 'time_signature':
            time_sigs.append((abs_tick, msg.numerator, msg.denominator))
    if not time_sigs:
        time_sigs.append((0, 4, 4))
    if time_sigs[0][0] != 0:
        time_sigs.insert(0, (0, 4, 4))

    # --- Build Bars ---
    bars = []  # Each bar: start, end, num, den, measure_ticks, total_grids, pickup flag, pickup_offset
    current_tick = 0
    ts_index = 0
    while current_tick < max_tick:
        # Update time signature: use the event at or before the current tick.
        if ts_index < len(time_sigs) and current_tick >= time_sigs[ts_index][0]:
            cur_ts = time_sigs[ts_index]
            ts_index += 1
        else:
            cur_ts = time_sigs[ts_index - 1] if ts_index > 0 else (0, 4, 4)
        num, den = cur_ts[1], cur_ts[2]

        # --- Adjust time signature to allowed final ones ---
        effective_ts_list = []
        if (num, den) in [(2,4), (3,4), (4,4), (3,8), (6,8)]:
            effective_ts_list.append((num, den))
        elif (num, den) == (5,4):
            effective_ts_list.extend([(2,4), (3,4)])
        elif (num, den) == (6,4):
            effective_ts_list.extend([(3,4), (3,4)])
        elif (num, den) == (12,8):
            effective_ts_list.extend([(6,8), (6,8)])
        else:
            effective_ts_list.append((4,4))
        
        # For each effective time signature (multiple bars if splitting is needed)
        for eff_num, eff_den in effective_ts_list:
            # Initialize measure_ticks and total_grids using the effective time signature.
            if eff_den == 4:
                measure_ticks = eff_num * 480
                total_grids = eff_num * 12
            elif eff_den == 8:
                measure_ticks = eff_num * 240
                total_grids = eff_num * 6
            else:
                measure_ticks = eff_num * 480
                total_grids = eff_num * 12

            # Compute original measure length from the original time signature event.
            if den == 4:
                orig_ticks = num * 480
                orig_grids = num * 12
            elif den == 8:
                orig_ticks = num * 240
                orig_grids = num * 6
            else:
                orig_ticks = num * 480
                orig_grids = num * 12

            is_pickup = False
            pickup_offset = 0

            # --- Revised Pickup/Incompleted Bar Handling ---
            # If the first bar is incomplete (pickup) then override its time signature with that of the next full bar.
            # Here we check if the original measure (pickup) is shorter than a full 4/4 measure (1920 ticks).
            if current_tick == 0 and orig_ticks < (480 * 4):
                # Look ahead to the next time signature event (if available)
                if ts_index < len(time_sigs):
                    next_ts = time_sigs[ts_index]
                    next_num, next_den = next_ts[1], next_ts[2]
                else:
                    next_num, next_den = 4, 4

                # Compute full measure length under the next time signature
                if next_den == 4:
                    full_ticks_next = next_num * 480
                elif next_den == 8:
                    full_ticks_next = next_num * 240
                else:
                    full_ticks_next = 4 * 480

                # If the pickup length is less than a full measure in the next time signature, treat as pickup.
                if orig_ticks < full_ticks_next:
                    is_pickup = True
                    # Override effective time signature with the next one for consistent padding.
                    eff_num, eff_den = next_num, next_den
                    # Adjust to allowed final time signatures if needed.
                    if (eff_num, eff_den) in [(2, 4), (3, 4), (4, 4), (3, 8), (6, 8)]:
                        pass
                    elif (eff_num, eff_den) == (5, 4):
                        eff_num, eff_den = 3, 4
                    elif (eff_num, eff_den) == (6, 4):
                        eff_num, eff_den = 3, 4
                    elif (eff_num, eff_den) == (12, 8):
                        eff_num, eff_den = 6, 8
                    else:
                        eff_num, eff_den = 4, 4

                    if eff_den == 4:
                        measure_ticks = eff_num * 480
                        total_grids = eff_num * 12
                    elif eff_den == 8:
                        measure_ticks = eff_num * 240
                        total_grids = eff_num * 6

                    # Padding: shift notes to the right by the missing grids.
                    pickup_offset = total_grids - orig_grids
                    bar_end = current_tick + orig_ticks
                else:
                    bar_end = current_tick + measure_ticks
            else:
                bar_end = current_tick + measure_ticks

            bars.append({
                'start': current_tick,
                'end': bar_end,
                'num': eff_num,
                'den': eff_den,
                'measure_ticks': measure_ticks,
                'total_grids': total_grids,
                'pickup': is_pickup,
                'pickup_offset': pickup_offset
            })
            current_tick = bar_end
            if current_tick >= max_tick:
                break
        if current_tick >= max_tick:
            break

    # --- Assign Note Events to Bars ---
    bars_notes = []
    for bar in bars:
        notes_in_bar = []
        for note in note_events:
            if bar['start'] <= note['start'] < bar['end']:
                local_tick = note['start'] - bar['start']
                if bar['pickup']:
                    # For pickup bars, use the actual (short) pickup length and apply padding offset.
                    pickup_length = bar['end'] - bar['start']
                    grid = int((local_tick / pickup_length) * (bar['total_grids'] - bar['pickup_offset'])) + bar['pickup_offset']
                    allowed_duration = pickup_length
                else:
                    grid = int((local_tick / bar['measure_ticks']) * bar['total_grids'])
                    allowed_duration = bar['measure_ticks']
                # Clamp grid (0-indexed)
                if grid >= bar['total_grids']:
                    grid = bar['total_grids'] - 1
                note_copy = note.copy()
                note_copy['grid'] = grid
                # Quantize duration: smallest unit is 40 ticks for denominator 4, 80 ticks for denominator 8.
                smallest_unit = 40 if bar['den'] == 4 else 80
                quantized_duration = round(note_copy['duration'] / smallest_unit) * smallest_unit
                quantized_duration = max(quantized_duration, smallest_unit)
                note_copy['duration'] = min(quantized_duration, int(allowed_duration))
                notes_in_bar.append(note_copy)
        # Group notes by grid position.
        grid_dict = {}
        for note in notes_in_bar:
            g = note['grid']
            grid_dict.setdefault(g, []).append(note)
        # Sort groups: within each grid, sort notes by descending pitch.
        for g in grid_dict:
            grid_dict[g].sort(key=lambda n: n['pitch'], reverse=True)
        grouped_notes = sorted(grid_dict.items(), key=lambda x: x[0])
        bars_notes.append(grouped_notes)

    # --- Build the Final Sequence ---
    sequence = []
    bar_indices = []

    # Global tokens: Composer, Tempo, BOS.
    sequence.append(create_event('Composer', composer))
    sequence.append(create_event('Tempo', global_tempo))
    sequence.append(create_event('BOS', None))

    # Process each bar: add Bar and Time_Signature tokens, then Beat and note events.
    for i, bar in enumerate(bars):
        bar_idx = len(sequence)
        sequence.append(create_event('Bar', None))
        bar_indices.append(bar_idx)
        ts_str = f"{bar['num']}/{bar['den']}"
        sequence.append(create_event('Time_Signature', ts_str))
        for grid, notes in bars_notes[i]:
            sequence.append(create_event('Beat', grid))
            for note in notes:
                if note['pitch'] > 108 or note['pitch'] < 21: 
                    return None, None
                sequence.append(create_event('Note_Pitch', note['pitch']))
                sequence.append(create_event('Note_Duration', int(note['duration'])))
    sequence.append(create_event('EOS', None))

    return sequence, bar_indices


if __name__ == '__main__':
    """
    convert midi to events
    """
    data_root = '../dataset/raw_data/huggingface_data'
    event_root = '../score_data/huggingface_data'
    add_composer = True
    os.makedirs(event_root, exist_ok=True)
    print('save dir:', event_root)

    for category in os.listdir(data_root):
        if category == '.DS_Store': continue
        midi_category_root = os.path.join(data_root, category)
        event_category_root = os.path.join(event_root, category)
        os.makedirs(event_category_root, exist_ok=True)

        midi_files = os.listdir(midi_category_root)
        print('Processing:', category)
        for file in tqdm(midi_files):
            filename = file[:-4] if (file.endswith('.mid') or file.endswith('MID')) else file[:-5]
            midi_path = os.path.join(midi_category_root, file)

            composer_name = 'None'
            if add_composer:
                composer_name = category.capitalize()

            try:
                events, bar_indices = \
                                midi_to_sequence(midi_path, composer_name)
                if events is None: continue
            except:
                continue
            pickle.dump((events, bar_indices),
                        open(os.path.join(event_category_root, filename + '.pkl'), 'wb'))

        print(f'{len(os.listdir(event_category_root))} files are good in {category}')