import pretty_midi
import numpy as np
from custom_music_generation.full_song.engine.key_engine import get_allowed_pcs


def snap_to_nearest_pc(pitch, allowed_pcs):
    """
    将 pitch 吸附到最近的合法音级，尽量保持旋律方向
    """
    base_octave = (pitch // 12) * 12
    candidates = []

    for pc in allowed_pcs:
        for shift in (-12, 0, 12):
            candidates.append(base_octave + pc + shift)

    # 优先：距离最小，其次：不上跳
    candidates.sort(key=lambda x: (abs(x - pitch), x < pitch))
    return candidates[0]


def apply_pitch_constraints(
    midi,
    tonic,
    mode,
    low=48,
    high=84,
    pentatonic=True
):
    """
    midi: pretty_midi.PrettyMIDI
    tonic_pc: 0–11
    mode: 'major' | 'minor'
    """

    allowed_pcs = get_allowed_pcs(
        tonic,
        mode,
        pentatonic=pentatonic
    )

    for inst in midi.instruments:
        if inst.is_drum:
            continue

        for note in inst.notes:
            # 音域限制
            while note.pitch < low:
                note.pitch += 12
            while note.pitch > high:
                note.pitch -= 12

            # 音级吸附
            if note.pitch % 12 not in allowed_pcs:
                note.pitch = snap_to_nearest_pc(
                    note.pitch,
                    allowed_pcs
                )

    return midi
