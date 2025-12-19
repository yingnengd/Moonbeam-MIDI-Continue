from pathlib import Path
import pretty_midi

from custom_music_generation.full_song.engine.scale_engine import get_allowed_pitch_classes
from custom_music_generation.full_song.engine.pitch_constraints import apply_pitch_constraints


def compose_section(
    section: str,
    bars: int,
    model,
    tonic: int,
    mode: str,
    hook,
    prompt_midi=None,
):
    """
    Generate one song section and return PrettyMIDI.
    """

    # =========================
    # 0️⃣ prompt_midi 防御（非常重要）
    # =========================
    if isinstance(prompt_midi, str):
        if not prompt_midi.endswith(".mid"):
            prompt_midi = None

    # =========================
    # 1️⃣ 段落 → 音阶策略
    # =========================
    if section in ["intro", "verse"]:
        scale_name = "minor_pentatonic" if mode == "minor" else "pentatonic"
    else:
        scale_name = mode

    allowed_pcs = get_allowed_pitch_classes(tonic, scale_name)

    # =========================
    # 2️⃣ 调用 MusicLLaMA（唯一正确方式）
    # =========================
    midi_result = model.generate(
        prompt_midi=prompt_midi,
        bars=bars,
        constraints={
            "tonic": tonic,
            "mode": mode,
            "section": section,
            "allowed_pcs": allowed_pcs,
        }
    )

    # =========================
    # 3️⃣ 统一成 PrettyMIDI
    # =========================
    if isinstance(midi_result, pretty_midi.PrettyMIDI):
        midi = midi_result
    elif isinstance(midi_result, str):
        if not Path(midi_result).exists():
            raise FileNotFoundError(midi_result)
        midi = pretty_midi.PrettyMIDI(midi_result)
    else:
        raise TypeError(f"Unexpected return type: {type(midi_result)}")

    # =========================
    # 4️⃣ 音域 + 调性约束
    # =========================
    midi = apply_pitch_constraints(
        midi,
        tonic=tonic,
        mode=mode,
        low=48,
        high=84,
        pentatonic=(scale_name in ["pentatonic", "minor_pentatonic"]),
    )

    # =========================
    # 5️⃣ Hook 逻辑（只存旋律）
    # =========================
    if section == "chorus":
        notes = []
        for inst in midi.instruments:
            if not inst.is_drum:
                notes.extend(inst.notes)
        notes.sort(key=lambda n: n.start)
        hook.store(notes)

    if section == "final" and hook.recall():
        motif_midi = pretty_midi.PrettyMIDI()
        instr = pretty_midi.Instrument(program=0)
        for n in hook.recall():
            instr.notes.append(
                pretty_midi.Note(n.velocity, n.pitch, n.start, n.end)
            )
        motif_midi.instruments.append(instr)
        motif_midi.instruments.extend(midi.instruments)
        midi = motif_midi

    return midi
