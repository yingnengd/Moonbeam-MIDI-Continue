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

SCALES = {
    "major":        [0, 2, 4, 5, 7, 9, 11],
    "minor":        [0, 2, 3, 5, 7, 8, 10],
    "pentatonic":   [0, 2, 4, 7, 9],
    "minor_penta":  [0, 3, 5, 7, 10],
    "dorian":       [0, 2, 3, 5, 7, 9, 10],
    "mixolydian":   [0, 2, 4, 5, 7, 9, 10],
}

def get_allowed_scale(tonic, scale_name="major"):
    return [(tonic + i) % 12 for i in SCALES[scale_name]]
