# ============================================================
# MusicLlama Local .pt | Full Song Continuation (Segmented)
# ============================================================

import os
import torch
import random
from pathlib import Path
from generation import MusicLlama

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

BARS_TO_TOKENS = 96  # 每小节对应 token 数
MAX_TOKENS = 1024    # 安全上限

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

def apply_jay_rules(midi_data, section, energy):
    """
    midi_data: MusicLlama decode 后的 midi 对象
    """
    if not midi_data.notes:
        return midi_data

    prev_degree = 1
    low, high = SECTION_PITCH_RANGE[energy]

    for note in midi_data.notes:
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

    end_degree = SECTION_END_DEGREE.get(section, 1)
    midi_data.notes[-1].pitch = TONIC + PENTATONIC[(end_degree - 1) % len(PENTATONIC)]

    return midi_data

# ============================================================
# LOAD LOCAL .PT MODEL (MusicLlama)
# ============================================================

MODEL_PT_PATH = Path("model_local.pt")  # <-- 改成你的权重路径
CONFIG_PATH   = Path("config.yaml")     # <-- 配置路径
TOKENIZER_PATH= Path("tokenizer.json")  # <-- tokenizer 路径

assert MODEL_PT_PATH.exists(), f"❌ Model not found: {MODEL_PT_PATH}"

generator = MusicLlama.build(
    ckpt_dir=str(MODEL_PT_PATH),
    model_config_path=str(CONFIG_PATH),
    tokenizer_path=str(TOKENIZER_PATH),
    max_seq_len=MAX_TOKENS,
    max_batch_size=1,
    finetuned_PEFT_weight_path=None,
).model.to(device)

generator.eval()
print("✅ MusicLlama local .pt loaded")

# ============================================================
# OUTPUT DIR
# ============================================================

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

previous_tokens = None
all_midis = []

# ============================================================
# GENERATION LOOP
# ============================================================

for idx, (section, bars, energy) in enumerate(SONG_STRUCTURE):
    print(f"🎼 Generating {section}")

    max_tokens = min(bars * BARS_TO_TOKENS, MAX_TOKENS)

    # prompt tokens 来自上一段生成
    prompts = [previous_tokens] if previous_tokens else []

    with torch.no_grad():
        results = generator.music_completion(
            prompt_tokens=prompts,
            temperature=ENERGY_CONFIG[energy]["temperature"],
            top_p=ENERGY_CONFIG[energy]["top_p"],
            max_gen_len=max_tokens,
        )

    # decode 生成的 midi
    generated_midi = results[0]["generation"]["content"]
    generated_midi = apply_jay_rules(generated_midi, section, energy)

    # 保存
    midi_path = output_dir / f"{idx:02d}_{section}.mid"
    generator.tokenizer.compound_to_midi(generated_midi).save(midi_path)

    previous_tokens = results[0]["tokens"]
    all_midis.append(midi_path)

# ============================================================
# CONCAT FULL SONG
# ============================================================

full_song = generator.tokenizer.compound_to_midi(
    sum([generator.tokenizer.encode_series(load_midi(p)) for p in all_midis], [])
)
full_song.save(output_dir / "FULL_SONG.mid")

print("🎉 DONE: outputs/FULL_SONG.mid")
