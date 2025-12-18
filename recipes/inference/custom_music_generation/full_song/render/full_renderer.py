# render/full_renderer.py
import pretty_midi
from pathlib import Path
from typing import List

def render_sections_full(
    base_midi: pretty_midi.PrettyMIDI,
    sections: List[pretty_midi.PrettyMIDI],
    out_dir: str = "outputs"
) -> pretty_midi.PrettyMIDI:
    """
    功能：
    1️⃣ 将 sections 拼接到 base_midi 上
    2️⃣ 输出每段 MIDI 文件和对应简谱
    3️⃣ 输出整首 MIDI 和整首简谱（旋律轨默认第一条）
    """
    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True, parents=True)

    full = pretty_midi.PrettyMIDI()
    # 拷贝原始 base_midi 的伴奏轨
    for inst in base_midi.instruments:
        full.instruments.append(inst)

    t_offset = base_midi.get_end_time()

    for i, midi in enumerate(sections):
        section_name = getattr(midi, "name", f"section_{i}")

        # ---- 导出单段 MIDI ----
        section_file = out_path / f"{i:02d}_{section_name}.mid"
        midi.write(section_file)

        # ---- 导出单段简谱（默认第一轨道为旋律） ----
        if midi.instruments:
            export_jianpu(midi.instruments[0], out_path / f"{i:02d}_{section_name}_jianpu.txt")

        # ---- 拼接到 full MIDI ----
        for inst in midi.instruments:
            new_inst = pretty_midi.Instrument(program=inst.program, is_drum=inst.is_drum)
            for n in inst.notes:
                new_inst.notes.append(
                    pretty_midi.Note(
                        velocity=n.velocity,
                        pitch=n.pitch,
                        start=n.start + t_offset,
                        end=n.end + t_offset
                    )
                )
            full.instruments.append(new_inst)

        t_offset += midi.get_end_time()

    # ---- 导出整首 MIDI ----
    full_file = out_path / "full_song.mid"
    full.write(full_file)

    # ---- 导出整首简谱（旋律轨默认第一条） ----
    if full.instruments:
        export_jianpu(full.instruments[0], out_path / "full_song_jianpu.txt")

    print(f"✅ 分段 MIDI + 简谱 + 整首 MIDI + 整首简谱 已导出到 {out_dir}")
    return full


def export_jianpu(inst: pretty_midi.Instrument, out_file: Path):
    """
    导出简谱：音高 -> 音级 (1~7)
    只处理传入的旋律轨
    """
    lines = []
    tonic = 60  # 默认 C4，可改为自动 key
    scale = [0, 2, 4, 5, 7, 9, 11]  # C大调音阶

    for n in inst.notes:
        deg = (n.pitch - tonic) % 12
        if deg in scale:
            deg = scale.index(deg) + 1
        else:
            deg = "×"
        dur = round(n.end - n.start, 2)
        lines.append(f"{deg} ({dur})")

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
