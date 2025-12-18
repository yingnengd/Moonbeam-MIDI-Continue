import pretty_midi

def render_sections(prompt, sections):
    full = pretty_midi.PrettyMIDI()
    for sec in prompt.instruments:
        full.instruments.append(sec)

    for midi in sections:
        for inst in midi.instruments:
            full.instruments.append(inst)

    return full
