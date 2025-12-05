import os
import numpy as np
import pretty_midi

# Folder with Chopin piano pieces
MIDI_DIR = "Midi"
OUT_PATH = "data/chopin_sequences.npz"

# Quantization: seconds per time step
TIME_STEP = 0.125  # ~16th note at ~120 bpm

# Determining pitch range for piano
MIN_PITCH = 21   # A0
MAX_PITCH = 108  # C8

def midi_to_pitch_grid(midi_path, time_step):
    """Convert MIDI file into a 1D sequence of pitches (monophonic)."""
    pm = pretty_midi.PrettyMIDI(midi_path)

    # Merging all instruments into one note list
    all_notes = []
    for inst in pm.instruments:
        all_notes.extend(inst.notes)

    if not all_notes:
        return np.array([], dtype=int)

    # Sort notes by start time
    all_notes.sort(key=lambda n: n.start)

    # Determine total time and number of steps
    total_time = max(n.end for n in all_notes)
    num_steps = int(total_time / time_step) + 1

    # Initialize with "rest" token = -1
    pitch_seq = np.full(num_steps, -1, dtype=int)

    # For overlapping notes, keep highest pitch
    for note in all_notes:
        start_idx = int(note.start / time_step)
        end_idx = int(note.end / time_step)
        for t in range(start_idx, end_idx + 1):
            if 0 <= t < num_steps:
                pitch = note.pitch
                if MIN_PITCH <= pitch <= MAX_PITCH:
                    if pitch_seq[t] < pitch:
                        pitch_seq[t] = pitch

    return pitch_seq

def build_dataset_from_folder(midi_dir, time_step):
    all_seqs = []
    for fname in os.listdir(midi_dir):
        if not fname.lower().endswith(".mid"):
            continue
        path = os.path.join(midi_dir, fname)
        print(f"Processing {path}...")
        seq = midi_to_pitch_grid(path, time_step)
        if len(seq) == 0:
            continue
        all_seqs.append(seq)

    if not all_seqs:
        raise RuntimeError("No valid sequences created from MIDI files.")

    # Concatenated sequences with a small rest gap between pieces
    gap = np.full(16, -1, dtype=int)  # 16 time steps of rest
    full_seq = all_seqs[0]
    for seq in all_seqs[1:]:
        full_seq = np.concatenate([full_seq, gap, seq])

    return full_seq

def save_sequences():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    full_seq = build_dataset_from_folder(MIDI_DIR, TIME_STEP)
    print(f"Full sequence length: {len(full_seq)} time steps")

    np.savez_compressed(OUT_PATH, pitch_sequence=full_seq)
    print(f"Saved pitch sequence to {OUT_PATH}")

if __name__ == "__main__":
    save_sequences()
