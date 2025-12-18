import pretty_midi

def render_sections(prompt, sections):
    full = pretty_midi.PrettyMIDI()
    for sec in prompt.instruments:
        full.instruments.append(sec)

    for midi in sections:
        for inst in midi.instruments:
            full.instruments.append(inst)

    return full

def export_jianpu_auto_key(midi: pretty_midi.PrettyMIDI, out_file: str):
    """
    根据 MIDI 调性自动生成简谱 (旋律轨道)
    """
    tonic, mode = extract_key_from_midi(midi)  # 返回 tonic(0-11), mode('major'/'minor')
    
    # 生成调式音阶半音偏移
    if mode == "major":
        scale = [0, 2, 4, 5, 7, 9, 11]  # 自然大调
    else:
        scale = [0, 2, 3, 5, 7, 8, 10]  # 自然小调

    lines = []

    # 只取旋律轨道（可以假设第 0 轨是旋律）
    melody_instr = midi.instruments[0] if midi.instruments else None
    if melody_instr is None:
        print("No instruments found in MIDI")
        return

    for n in melody_instr.notes:
        # 将 MIDI 音高映射到调式音级
        deg = (n.pitch - tonic - 60) % 12  # MIDI 60=C4 对齐
        if deg in scale:
            deg = scale.index(deg) + 1  # 1~7
        else:
            deg = "×"  # 非调式音
        dur = round(n.end - n.start, 2)
        lines.append(f"{deg} ({dur})")

    out_path = Path(out_file)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✅ Simplified notation exported to {out_file}")
