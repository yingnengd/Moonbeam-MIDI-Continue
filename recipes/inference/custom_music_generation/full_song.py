# ============================================================
# MusicLlama Local .pt | Full Song Continuation (Segmented)
# ============================================================

import os
import torch
import random
from pathlib import Path
from generation import MusicLlama# ============================================================
# MusicLlama Local .pt | Full Song Continuation (NO CONFIG)
# ============================================================

import pretty_midi


# ============================================================
# TORCH / DEVICE
# ============================================================

torch.set_float32_matmul_precision("high")

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"🚀 Using device: {device}")

# ============================================================
# SONG STRUCTURE
# ============================================================

SONG_STRUCTURE = [
    ("intro",   8,  "low"),
    ("verse",   16, "low"),
    ("pre",     8,  "mid"),
    ("chorus",  8,  "high"),
    ("verse",   16, "low"),
    ("pre",     8,  "mid"),
    ("chorus",  8,  "high"),
    ("bridge",  8,  "low"),
    ("final",   12, "high"),
    ("outro",   4,  "low"),
]

ENERGY_CONFIG = {
    "low":  dict(temperature=0.8,  top_p=0.9),
    "mid":  dict(temperature=1.0,  top_p=0.9),
    "high": dict(temperature=1.15, top_p=0.95),
}

BARS_TO_TOKENS = 96
MAX_TOKENS = 1024

# ============================================================
# JAY CHOU STYLE RULES
# ============================================================

TONIC = 60
PENTATONIC = [0, 2, 4, 7, 9]

MELODY_TRANSITION = {
    1: [(2,0.4),(3,0.4),(5,0.2)],
    2: [(3,0.6),(5,0.4)],
    3: [(5,0.8),(6,0.2)],
    5: [(6,0.5),(3,0.3),(1,0.2)],
    6: [(5,0.6),(1,0.4)],
}

SECTION_END_DEGREE = {
    "intro": 1,
    "verse": 1,
    "pre": 3,
    "chorus": 5,
    "bridge": 1,
    "final": 5,
    "outro": 1,
}

SECTION_PITCH_RANGE = {
    "low":  (60, 72),
    "mid":  (62, 76),
    "high": (67, 81),
}

def jay_next_degree(prev):
    choices, weights = zip(*MELODY_TRANSITION.get(prev, [(prev, 1.0)]))
    return random.choices(choices, weights)[0]

def apply_jay_rules(midi, section, energy):
    if not hasattr(midi, "notes"):
        return midi

    prev_degree = 1
    low, high = SECTION_PITCH_RANGE[energy]

    for n in midi.notes:
        degree = jay_next_degree(prev_degree)
        pitch = TONIC + PENTATONIC[(degree - 1) % 5]

        while pitch < low:
            pitch += 12
        while pitch > high:
            pitch -= 12

        n.pitch = pitch

        if energy == "high":
            n.velocity = min(127, int(n.velocity * 1.2))
        elif energy == "low":
            n.velocity = int(n.velocity * 0.85)

        prev_degree = degree

    end_deg = SECTION_END_DEGREE.get(section, 1)
    midi.notes[-1].pitch = TONIC + PENTATONIC[(end_deg - 1) % 5]
    return midi


# ============================================================
# CONCAT ALL SECTIONS INTO FULL SONG
# ============================================================

def concat_midis(midi_paths, out_path):
    full = pretty_midi.PrettyMIDI()
    instrument_map = {}
    current_time = 0.0

    for p in midi_paths:
        m = pretty_midi.PrettyMIDI(str(p))

        for inst in m.instruments:
            if inst.program not in instrument_map:
                new_inst = pretty_midi.Instrument(program=inst.program)
                full.instruments.append(new_inst)
                instrument_map[inst.program] = new_inst

            target_inst = instrument_map[inst.program]

            for note in inst.notes:
                target_inst.notes.append(
                    pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=note.start + current_time,
                        end=note.end + current_time
                    )
                )

        # 推进时间轴
        current_time += m.get_end_time()

    full.write(str(out_path))
    

# ============================================================
# LOAD LOCAL PT MODEL (NO CONFIG)
# ============================================================

MODEL_PT_PATH = Path("../../../../moonbeam-model/moonbeam_839M.pt")  # ← 改成你的 pt 路径
assert MODEL_PT_PATH.exists(), "❌ model pt not found"

print("📦 Loading model...")
bundle = torch.load(MODEL_PT_PATH, map_location=device)

# 兼容不同保存方式
model = bundle.get("model", bundle)
tokenizer = bundle.get("tokenizer", None)

model = model.to(device).eval()

print("✅ Model loaded")

# ============================================================
# OUTPUT
# ============================================================

out_dir = Path("outputs")
out_dir.mkdir(exist_ok=True)

prev_tokens = None
all_midis = []

# ============================================================
# GENERATION LOOP
# ============================================================

for i, (section, bars, energy) in enumerate(SONG_STRUCTURE):
    print(f"🎼 {section}")

    max_len = min(bars * BARS_TO_TOKENS, MAX_TOKENS)

    with torch.no_grad():
        if hasattr(model, "music_completion"):
            out = model.music_completion(
                prompt_tokens=[prev_tokens] if prev_tokens else None,
                temperature=ENERGY_CONFIG[energy]["temperature"],
                top_p=ENERGY_CONFIG[energy]["top_p"],
                max_gen_len=max_len,
            )
            tokens = out[0]["tokens"]
            midi = out[0]["generation"]["content"]

        elif hasattr(model, "generate"):
            tokens = model.generate(
                prompt_tokens=prev_tokens,
                max_length=max_len,
                temperature=ENERGY_CONFIG[energy]["temperature"],
                top_p=ENERGY_CONFIG[energy]["top_p"],
            )
            midi = tokenizer.decode(tokens)

        else:
            raise RuntimeError("❌ Unknown MusicLlama PT interface")

    midi = apply_jay_rules(midi, section, energy)

    path = out_dir / f"{i:02d}_{section}.mid"
    midi.save(path)

    prev_tokens = tokens
    all_midis.append(path)

# ============================================================
# DONE
# ============================================================

print("🎉 Part_MIDI GENERATED")




# 执行拼接
full_song_path = out_dir / "FULL_SONG.mid"
concat_midis(all_midis, full_song_path)

print(f"🎧 FULL SONG READY → {full_song_path}")



