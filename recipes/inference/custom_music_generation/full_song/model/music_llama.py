import sys

#from generation import MusicLlama
#from custom_music_generation.generation import Llama

from custom_music_generation.generation import MusicLlama
from model.base_model import BaseMusicModel



from model.base_model import BaseMusicModel

class MusicLlamaModel(BaseMusicModel):
    def __init__(self, ckpt):
        self.model = MusicLlama.load_from_checkpoint(ckpt)

    def generate(self, prompt_midi, bars, constraints):
        return self.model.generate(
            prompt_midi=prompt_midi,
            max_bars=bars,
            constraints=constraints,
            temperature=0.9
        )
