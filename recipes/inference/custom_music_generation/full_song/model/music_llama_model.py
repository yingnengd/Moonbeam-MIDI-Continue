import pretty_midi
from custom_music_generation.full_song.model.base_model import BaseMusicModel
from custom_music_generation.full_song.model.music_llama import MusicLLaMA


class MusicLlamaModel(BaseMusicModel):
    """
    Adapter for Moonbeam / MusicLLaMA

    Contract:
    - generate() ALWAYS returns PrettyMIDI
    """

    def __init__(self, ckpt_path, device="mps"):
        self.model = MusicLLaMA(
            ckpt_path=ckpt_path,
            device=device
        )

    def generate(self, prompt_midi, bars, constraints=None):
        """
        Generate MIDI continuation.

        Returns:
            pretty_midi.PrettyMIDI
        """

        result = self.model.generate(
            prompt_midi=prompt_midi,
            bars=bars,
            constraints=constraints,
        )

        # ✅ 正确情况
        if isinstance(result, pretty_midi.PrettyMIDI):
            return result

        # ❌ 一律不兜底、不猜、不扫目录
        raise RuntimeError(
            f"MusicLLaMA.generate() did not return PrettyMIDI, got: {type(result)} | {result}"
        )
