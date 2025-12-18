from custom_music_generation.full_song.model.music_llama import MusicLLaMA
from custom_music_generation.full_song.model.base_model import BaseMusicModel

class MusicLlamaModel(BaseMusicModel):
    def __init__(self, ckpt_path, device="mps"):
        self.model = MusicLLaMA(
            ckpt_path=ckpt_path,
            device=device
        )

    def generate(self, prompt_midi, bars, constraints):
        # 参数名对齐 + 转发
        return self.model.generate(
            prompt_midi=prompt_midi,
            n_bars=bars,
            constraints=constraints,
            temperature=0.9,
            top_p=0.95
        )
