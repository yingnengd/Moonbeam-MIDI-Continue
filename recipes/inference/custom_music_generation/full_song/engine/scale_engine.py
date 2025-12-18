def get_allowed_scale(section, mode):
    if section == "verse":
        return [0,2,4,7,9] if mode == "major" else [0,3,5,7,10]
    else:
        return (
            [0,2,4,5,7,9,11]
            if mode == "major"
            else [0,2,3,5,7,8,10]
        )
