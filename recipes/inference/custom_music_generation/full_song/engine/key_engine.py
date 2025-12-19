# engine/key_engine.py

import numpy as np

KEY_NAMES = ["C", "C#", "D", "D#", "E", "F",
             "F#", "G", "G#", "A", "A#", "B"]

def extract_key_from_midi(pm):
    """
    返回:
        tonic_pc: 0–11
        mode: "major" | "minor"
    """
    pitch_class_hist = np.zeros(12)

    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            pitch_class_hist[n.pitch % 12] += (n.end - n.start)

    if pitch_class_hist.sum() == 0:
        return 0, "major"

    tonic_pc = int(np.argmax(pitch_class_hist))

    major_third = pitch_class_hist[(tonic_pc + 4) % 12]
    minor_third = pitch_class_hist[(tonic_pc + 3) % 12]

    mode = "major" if major_third >= minor_third else "minor"

    print(f"🎼 Detected key: {KEY_NAMES[tonic_pc]} {mode}")

    return tonic_pc, mode


def get_allowed_pcs(tonic_pc, mode, pentatonic=True):
    """
    根据调性返回允许的音级（pitch class）
    """

    if pentatonic:
        if mode == "major":
            scale = [0, 2, 4, 7, 9]
        else:
            scale = [0, 3, 5, 7, 10]
    else:
        if mode == "major":
            scale = [0, 2, 4, 5, 7, 9, 11]
        else:
            scale = [0, 2, 3, 5, 7, 8, 10]

    return {(tonic_pc + pc) % 12 for pc in scale}
