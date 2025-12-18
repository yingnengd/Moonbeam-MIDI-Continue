import pretty_midi
from engine.song_structure import SECTION_PLAN
from engine.key_engine import extract_key_from_midi
from engine.hook_memory import HookMemory
from composer.compose_section import compose_section
from render.midi_renderer import render_sections_full
from model.music_llama import MusicLlamaModel

prompt = pretty_midi.PrettyMIDI("prompt.mid")
tonic, mode = extract_key_from_midi(prompt)

model = MusicLlamaModel("ckpt/music_llama")
hook = HookMemory()

sections = []
current_prompt = prompt

for section, bars in SECTION_PLAN:
    midi = compose_section(
        section,
        bars,
        tonic,
        mode,
        hook,
        model,
        current_prompt
    )
    sections.append(midi)
    current_prompt = midi  # 真·续写

full = render_sections_full(prompt, sections, out_dir="outputs")
