'''
from custom_music_generation.full_song.engine.scale_engine import get_allowed_scale
from custom_music_generation.full_song.engine.modulation_engine import apply_modulation

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
'''
#------------

from custom_music_generation.full_song.engine.scale_engine import get_allowed_scale
from custom_music_generation.full_song.engine.modulation_engine import apply_modulation


def compose_section(
    section,
    bars,
    tonic,
    mode,
    hook,
    model,
    prompt_midi
):
    # 🎼 段落 → 音阶
    if section in ["intro", "verse"]:
        scale_name = "minor_pentatonic" if mode == "minor" else "pentatonic"
    elif section == "chorus":
        scale_name = mode
    else:
        scale_name = mode

    allowed_pcs = get_allowed_pitch_classes(tonic, scale_name)

    # 🎹 生成旋律想法
    midi = model.generate(
        prompt_midi=prompt_midi,
        bars=bars,
        constraints={
            "tonic": tonic,
            "mode": mode
        }
    )

    # 🔧 校音（关键一步）
    midi = apply_pitch_constraints(midi, allowed_pcs)

    if section == "chorus":
        hook.store(midi)

    if section == "final" and hook.recall():
        midi = hook.recall() + midi

    return midi

