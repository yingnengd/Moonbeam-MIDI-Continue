import sys
from pathlib import Path

# 将上一级目录加入 sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from generation import MusicLlama


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
