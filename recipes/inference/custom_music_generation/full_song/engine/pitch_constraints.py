import pretty_midi
import numpy as np

def snap_to_nearest_pc(pitch, allowed_pcs):
    candidates = []
    for pc in allowed_pcs:
        for octave in range(-2, 3):
            candidates.append(pc + 12 * octave)
    return min(candidates, key=lambda x: abs(x - pitch))

def apply_pitch_constraints(midi, allowed_pcs, low=48, high=84):
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            while note.pitch < low:
                note.pitch += 12
            while note.pitch > high:
                note.pitch -= 12

            if note.pitch % 12 not in allowed_pcs:
                note.pitch = snap_to_nearest_pc(note.pitch, allowed_pcs)
    return midi
