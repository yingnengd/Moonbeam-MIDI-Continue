'''
import pretty_midi
from engine.song_structure import SECTION_PLAN
from engine.key_engine import extract_key_from_midi
from engine.hook_memory import HookMemory
from composer.compose_section import compose_section
from render.midi_renderer import render_sections_full
from model.music_llama import MusicLlamaModel
'''
from pathlib import Path
import pretty_midi

BASE_DIR = Path(__file__).resolve().parent
PROMPT_MIDI = BASE_DIR / "promptMIDI" / "prompt.mid"

import os

from custom_music_generation.full_song.engine.song_structure import SECTION_PLAN
from custom_music_generation.full_song.engine.key_engine import extract_key_from_midi
from custom_music_generation.full_song.engine.hook_memory import HookMemory

from custom_music_generation.full_song.composer.compose_section import compose_section
from custom_music_generation.full_song.render.midi_renderer import render_sections_full
from custom_music_generation.full_song.model.music_llama import MusicLlamaModel

prompt = pretty_midi.PrettyMIDI(str(PROMPT_MIDI))
tonic, mode = extract_key_from_midi(prompt)

model = MusicLlamaModel("../../../../moonbeam-model/moonbeam_839M.pt")
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
