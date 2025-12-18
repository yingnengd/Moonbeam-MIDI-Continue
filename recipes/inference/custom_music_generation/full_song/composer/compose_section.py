from engine.scale_engine import get_allowed_scale
from engine.modulation_engine import apply_modulation

def compose_section(section, bars, tonic, mode, hook, model, prompt):
    local_tonic = apply_modulation(section, tonic)
    scale = get_allowed_scale(section, mode)

    constraints = {
        "tonic": local_tonic,
        "scale": scale,
        "vocal_range": (60, 80)
    }

    midi = model.generate(
        prompt_midi=prompt,
        bars=bars,
        constraints=constraints
    )

    if section == "chorus":
        hook.store(midi)

    if section == "final" and hook.recall():
        midi = hook.recall() + midi

    return midi
