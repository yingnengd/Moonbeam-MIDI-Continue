'''
def extract_key_from_midi(pm):
    # 分析音高分布
    # 返回 tonic (60=C), mode ("major"/"minor")
    return tonic, mode
'''

import numpy as np

# pitch class → key 名
KEY_NAMES = ["C", "C#", "D", "D#", "E", "F",
             "F#", "G", "G#", "A", "A#", "B"]

def extract_key_from_midi(pm):
    """
    从 MIDI 中估计调性
    返回:
        tonic: int (60 = C4)
        mode: "major" | "minor"
    """

    pitch_class_hist = np.zeros(12)

    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            pitch_class_hist[n.pitch % 12] += (n.end - n.start)

    if pitch_class_hist.sum() == 0:
        # fallback：默认 C 大调
        return 60, "major"

    tonic_pc = int(np.argmax(pitch_class_hist))

    # 简单判断大小调：
    # 大三度 vs 小三度
    major_third = pitch_class_hist[(tonic_pc + 4) % 12]
    minor_third = pitch_class_hist[(tonic_pc + 3) % 12]

    mode = "major" if major_third >= minor_third else "minor"

    tonic_midi = 60 + tonic_pc  # 以 C4 为基准

    print(f"🎼 Detected key: {KEY_NAMES[tonic_pc]} {mode}")

    return tonic_midi, mode

