def apply_modulation(section, tonic):
    if section == "final":
        return tonic + 12
    if section == "chorus":
        return tonic + 2
    return tonic
