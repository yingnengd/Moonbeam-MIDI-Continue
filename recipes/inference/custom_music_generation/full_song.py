import os
os.environ["DISABLE_FLASH_ATTN"] = "1"

import torch
import random
from pathlib import Path
from moonbeam import Moonbeam
from moonbeam.midi import load_midi, save_midi, concat_midis

# =========================
# TORCH SETUP
# =========================
torch.set_float32_matmul_precision("high")

# =========================
# DEVICE (MAC / CUDA SAFE)
# =========================
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"🚀 Using device: {device}")

# =========================
# SONG STRUCTURE
# =========================
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
    ("outro",   4,  "low")
]

ENERGY_CONFIG = {
    "low":  dict(temperature=0.8,  top_k=40),
    "mid":  dict(temperature=1.0,  top_k=60),
    "high": dict(temperature=1.15, top_k=90),
}

BARS_TO_TOKENS = 96
MAX_TOKENS = 1024   # Moonbeam 安全上限

# =========================
# JAY STYLE RULES（旋律规则）
# =========================
TONIC = 60  # C
PENTATONIC = [0, 2, 4, 7, 9]  # 五声音阶

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
    "outro": 1
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
    if not midi.notes:
        return midi

    prev_degree = 1
    low, high = SECTION_PITCH_RANGE[energy]

    for note in midi.notes:
        degree = jay_next_degree(prev_degree)
        pitch = TONIC + PENTATONIC[(degree - 1) % len(PENTATONIC)]

        while pitch < low:
            pitch += 12
        while pitch > high:
            pitch -= 12

        note.pitch = pitch

        if energy == "high":
            note.velocity = min(127, int(note.velocity * 1.2))
        elif energy == "low":
            note.velocity = int(note.velocity * 0.85)

        prev_degree = degree

    # 段落收尾音
    end_degree = SECTION_END_DEGREE.get(section, 1)
    midi.notes[-1].pitch = TONIC + PENTATONIC[(end_degree - 1) % len(PENTATONIC)]

    return midi

# =========================
# LOAD LOCAL .pt MODEL
# =========================
MODEL_PT_PATH = Path("../../../../moonbeam-model/moonbeam-model.pt")

assert MODEL_PT_PATH.exists(), f"❌ Model not found: {MODEL_PT_PATH}"

model = Moonbeam()

state = torch.load(MODEL_PT_PATH, map_location=device)

# 兼容不同保存格式
if isinstance(state, dict) and "state_dict" in state:
    model.load_state_dict(state["state_dict"], strict=False)
else:
    model.load_state_dict(state, strict=False)

model = model.to(device)
model.eval()

print("✅ Moonbeam local .pt loaded")

# =========================
# OUTPUT DIR
# =========================
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

# =========================
# GENERATION LOOP
# =========================
previous_midi = None
all_midis = []

for idx, (section, bars, energy) in enumerate(SONG_STRUCTURE):
    print(f"🎼 Generating {section}")

    max_tokens = min(bars * BARS_TO_TOKENS, MAX_TOKENS)
    seed = load_midi(previous_midi) if previous_midi else None

    with torch.no_grad():
        generated = model.generate(
            seed=seed,
            max_new_tokens=max_tokens,
            temperature=ENERGY_CONFIG[energy]["temperature"],
            top_k=ENERGY_CONFIG[energy]["top_k"],
            use_cache=False
        )

    generated = apply_jay_rules(generated, section, energy)

    path = output_dir / f"{idx:02d}_{section}.mid"
    save_midi(generated, path)

    previous_midi = path
    all_midis.append(path)

# =========================
# CONCAT FULL SONG
# =========================
final_song = concat_midis(all_midis)
save_midi(final_song, output_dir / "FULL_SONG.mid")

print("🎉 DONE: outputs/FULL_SONG.mid")
