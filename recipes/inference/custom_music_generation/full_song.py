import os
os.environ["DISABLE_FLASH_ATTN"] = "1"

import torch
import random
from pathlib import Path
from moonbeam import Moonbeam
from moonbeam.midi import load_midi, save_midi, concat_midis

torch.set_float32_matmul_precision("high")

# =========================
# DEVICE (MAC SAFE)
# =========================
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

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

# =========================
# JAY STYLE RULES（不变）
# =========================
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
    "intro": 1, "verse": 1, "pre": 3,
    "chorus": 5, "bridge": 1, "final": 5, "outro": 1
}

SECTION_PITCH_RANGE = {
    "low":  (60, 72),
    "mid":  (62, 76),
    "high": (67, 81),
}

def jay_next_degree(prev):
    choices, weights = zip(*MELODY_TRANSITION.get(prev, [(prev,1.0)]))
    return random.choices(choices, weights)[0]

def apply_jay_rules(midi, section, energy):
    prev_degree = 1
    low, high = SECTION_PITCH_RANGE[energy]

    for note in midi.notes:
        degree = jay_next_degree(prev_degree)
        pitch = TONIC + PENTATONIC[(degree - 1) % len(PENTATONIC)]
        while pitch < low: pitch += 12
        while pitch > high: pitch -= 12
        note.pitch = pitch
        if energy == "high":
            note.velocity = min(127, int(note.velocity * 1.2))
        elif energy == "low":
            note.velocity = int(note.velocity * 0.85)
        prev_degree = degree

    midi.notes[-1].pitch = TONIC + PENTATONIC[(SECTION_END_DEGREE[section]-1)%5]
    return midi

# =========================
# MODEL
# =========================
model = Moonbeam.from_pretrained(
    "guozixunnicolas/moonbeam-midi-foundation-model",
    torch_dtype=torch.float32
).to(device)
model.eval()

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

# =========================
# GENERATION LOOP
# =========================
previous_midi = None
all_midis = []

for idx, (section, bars, energy) in enumerate(SONG_STRUCTURE):
    print(f"🎼 {section}")

    max_tokens = min(bars * BARS_TO_TOKENS, 1024)
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

save_midi(concat_midis(all_midis), output_dir / "FULL_SONG.mid")
print("✅ DONE")
