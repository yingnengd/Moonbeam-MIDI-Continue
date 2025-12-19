from pathlib import Path
import pretty_midi

BASE_DIR = Path(__file__).resolve().parent
PROMPT_MIDI = BASE_DIR / "promptMIDI" / "prompt.mid"

import os

from custom_music_generation.full_song.model.music_llama_model import MusicLlamaModel
import miditoolkit


from custom_music_generation.full_song.engine.song_structure import SECTION_PLAN
from custom_music_generation.full_song.engine.key_engine import extract_key_from_midi
from custom_music_generation.full_song.engine.hook_memory import HookMemory

from custom_music_generation.full_song.composer.compose_section import compose_section
from custom_music_generation.full_song.render.midi_renderer import render_sections_full
#from custom_music_generation.full_song.model.music_llama import MusicLLaMA
from custom_music_generation.full_song.engine.pitch_constraints import apply_pitch_constraints

# ✅ 从 pipeline 导入 MusicLLaMA
from custom_music_generation.full_song.model.music_llama_pipeline import MusicLLaMA

# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent
PROMPT_MIDI = BASE_DIR / "/Users/zideng/Moonbeam-MIDI-Continue/recipes/inference/custom_music_generation/full_song/promptMIDI/prompt.mid"

ckpt_path ="/Users/zideng/Moonbeam-MIDI-Continue/recipes/inference/custom_music_generation/full_song/moonbeam-model/moonbeam_839M.pt"

Path("/Users/zideng/Moonbeam-MIDI-Continue/recipes/inference/custom_music_generation/full_song/outputs").mkdir(exist_ok=True)

# 1️⃣ 定义输出路径
out_path = "/Users/zideng/Moonbeam-MIDI-Continue/recipes/inference/custom_music_generation/full_song/outputs/final_song.mid"

# =========================
# Load prompt
# =========================
print("🎵 Loading prompt MIDI...")
prompt = pretty_midi.PrettyMIDI(str(PROMPT_MIDI))

print("🎼 Detecting key...")
tonic, mode = extract_key_from_midi(prompt)
print(f"🎹 Key = {tonic % 12}, Mode = {mode}")


# =========================
# Load model
# =========================
print("🤖 Loading model...")
model = MusicLLaMA(ckpt_path=ckpt_path,device="mps")
hook = HookMemory()
# =========================
# Generate sections
# =========================
sections = []
current_prompt = prompt

# =========================
# 🎹 示例运行
# =========================
if __name__ == "__main__":
    model = MusicLLaMA(ckpt_path="/Users/zideng/Moonbeam-MIDI-Continue/recipes/inference/custom_music_generation/full_song/moonbeam-model/moonbeam_839M.pt", device="mps")

    for section_name, bars in SECTION_PLAN:
        print(f"🎶 Generating section: {section_name}")
        midi = compose_section(
            section=section_name,
            bars=bars,
            tonic=tonic,
            mode=mode,
            model=model,
            hook=hook,
            prompt_midi=current_prompt,
        )
        sections.append(midi)
        current_prompt = midi  # 真正续写


    # =========================
    # Render full song
    # =========================
    print("🎼 Rendering full song...")
    full_midi = render_sections_full(
        prompt,
        sections,
        out_dir="outputs"
    )


    # =========================
    # Post-process: key & range
    # =========================
    print("🎹 Applying pitch constraints...")
    full_midi = apply_pitch_constraints(
        full_midi,
        tonic=tonic % 12,
        mode=mode
    )

    # 保存文件
    # =========================
    if not full_midi.instruments:
        print("⚠️ Warning: full_midi has no instruments!")

    for inst in full_midi.instruments:
        print(f"Instrument {inst.program}, notes: {len(inst.notes)}")

    # 确保目录存在
    out_dir = Path(out_path).parent
    out_dir.mkdir(exist_ok=True)

    # 写文件
    full_midi.write(out_path)
    print(f"✅ Saved: {out_path}")