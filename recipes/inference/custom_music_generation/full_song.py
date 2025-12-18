import torch
import random
from pathlib import Path
from moonbeam import Moonbeam
from moonbeam.midi import load_midi, save_midi, concat_midis

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

BARS_TO_TOKENS = 96  # Moonbeam 默认

# =========================
# JAY STYLE RULES
# =========================
TONIC = 60  # C
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

# =========================
# MELODY POST PROCESS
# =========================
def jay_next_degree(prev):
    choices, weights = zip(*MELODY_TRANSITION.get(prev, [(prev,1.0)]))
    return random.choices(choices, weights)[0]

def apply_jay_rules(midi, section, energy):
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

    end_degree = SECTION_END_DEGREE.get(section, 1)
    midi.notes[-1].pitch = TONIC + PENTATONIC[(end_degree - 1) % len(PENTATONIC)]
    return midi

# =========================
# 🔧 NEW 1: BREATH MODELING
# =========================
def apply_breathing(midi, bars_per_phrase=2):
    if not midi.notes:
        return midi

    ticks_per_bar = midi.ticks_per_beat * 4
    phrase_ticks = bars_per_phrase * ticks_per_bar

    for i in range(len(midi.notes) - 1):
        cur = midi.notes[i]

        if cur.end % phrase_ticks < midi.ticks_per_beat:
            gap = random.randint(
                int(0.1 * midi.ticks_per_beat),
                int(0.25 * midi.ticks_per_beat)
            )
            cur.end -= gap

        dur = cur.end - cur.start
        if dur > midi.ticks_per_beat:
            cur.end = cur.start + int(dur * 0.88)

    return midi

# =========================
# 🔧 NEW 2: PRE-CHORUS INHALE BAR
# =========================
def apply_pre_chorus_breath(midi):
    if not midi.notes:
        return midi

    ticks_per_bar = midi.ticks_per_beat * 4
    last_bar_start = midi.notes[-1].start - ticks_per_bar

    for n in midi.notes:
        if n.start >= last_bar_start:
            n.velocity = int(n.velocity * 0.7)
            n.pitch -= random.choice([0, 2])
            dur = n.end - n.start
            n.end = n.start + int(dur * 0.75)

    midi.notes[-1].end = midi.notes[-1].start + midi.ticks_per_beat
    return midi

# =========================
# 🔧 NEW 3: JIANPU EXPORT
# =========================
DEGREE_MAP = {0:"1",2:"2",4:"3",5:"4",7:"5",9:"6",11:"7"}

def pitch_to_degree(pitch):
    return DEGREE_MAP.get((pitch - TONIC) % 12, "x")

def export_jianpu(midi, path):
    out = []
    line = []

    for n in midi.notes:
        deg = pitch_to_degree(n.pitch)
        octave = (n.pitch - TONIC) // 12
        mark = "'" * octave if octave > 0 else ""
        line.append(f"{deg}{mark}")

        if len(line) >= 8:
            out.append(" ".join(line))
            line = []

    if line:
        out.append(" ".join(line))

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))

# =========================
# INIT MODEL
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Moonbeam.from_pretrained(
    "guozixunnicolas/moonbeam-midi-foundation-model"
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
    print(f"🎼 Generating {section} ({bars} bars, {energy})")

    max_tokens = bars * BARS_TO_TOKENS
    params = ENERGY_CONFIG[energy]
    seed = load_midi(previous_midi) if previous_midi else None

    with torch.no_grad():
        generated = model.generate(
            seed=seed,
            max_new_tokens=max_tokens,
            temperature=params["temperature"],
            top_k=params["top_k"]
        )

    generated = apply_jay_rules(generated, section, energy)
    generated = apply_breathing(generated)

    if section == "pre":
        generated = apply_pre_chorus_breath(generated)

    section_path = output_dir / f"{idx:02d}_{section}.mid"
    save_midi(generated, section_path)

    export_jianpu(
        generated,
        output_dir / f"{idx:02d}_{section}_简谱.txt"
    )

    previous_midi = section_path
    all_midis.append(section_path)

# =========================
# CONCAT FINAL SONG
# =========================
final_song = concat_midis(all_midis)
save_midi(final_song, output_dir / "FULL_SONG.mid")

print("✅ DONE: FULL_SONG.mid")
