'''
def get_allowed_scale(section, mode):
    if section == "verse":
        return [0,2,4,7,9] if mode == "major" else [0,3,5,7,10]
    else:
        return (
            [0,2,4,5,7,9,11]
            if mode == "major"
            else [0,2,3,5,7,8,10]
        )
'''

# engine/scale_engine.py

SCALES = {
    "major":        [0, 2, 4, 5, 7, 9, 11],
    "minor":        [0, 2, 3, 5, 7, 8, 10],

    # 五声音阶（周杰伦常用）
    "pentatonic":       [0, 2, 4, 7, 9],
    "minor_pentatonic":[0, 3, 5, 7, 10],

    # 调式（进阶）
    "dorian":      [0, 2, 3, 5, 7, 9, 10],
    "mixolydian":  [0, 2, 4, 5, 7, 9, 10],
    "lydian":      [0, 2, 4, 6, 7, 9, 11],
}

def get_allowed_pitch_classes(
    tonic_midi: int,
    scale_name: str
):
    tonic = tonic_midi % 12
    return [(tonic + i) % 12 for i in SCALES[scale_name]]

