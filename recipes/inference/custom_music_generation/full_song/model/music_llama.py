# music_llama.py
import torch
from pathlib import Path
import pretty_midi
from datetime import datetime

# ✅ 关键：导入 Moonbeam / MIDI-LLM 包装层
from .moonbeam_wrapper import MoonbeamWrapper


class MusicLLaMA:
    """
    MusicLLaMA = 高层音乐生成 Pipeline
    - 负责 MIDI 读写
    - 负责约束传递
    - 调用 MoonbeamWrapper 做真正的 token 生成
    """

    def __init__(self, ckpt_path, model,device="mps"):
        # -------- device 处理（Mac / CUDA / CPU）--------
        if device == "mps" and not torch.backends.mps.is_available():
            device = "cpu"
        self.device = torch.device(device)

        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        # -------- 初始化真正的模型 wrapper --------
        self.model = MoonbeamWrapper(
            ckpt_path=ckpt_path,
            device=self.device
        )

        print(f"✅ MusicLLaMA initialized on {self.device}")
        print(f"✅ Using checkpoint: {ckpt_path}")

    # =========================================================
    # 🔮 对外主入口（compose_section / main.py 调用的就是它）
    # =========================================================
    def generate(
        self,
        prompt_midi=None,
        bars=8,
        constraints=None,
        temperature=1.0,
        top_p=0.9,
        save=True
    ):
        """
        prompt_midi: str | PrettyMIDI | None
        bars: 生成小节数
        constraints: 调性 / 音阶 / 音域 / 情绪等
        """

        # ---------- 1️⃣ 准备 prompt MIDI ----------
        if isinstance(prompt_midi, str) and Path(prompt_midi).exists():
            midi = pretty_midi.PrettyMIDI(prompt_midi)
        elif isinstance(prompt_midi, pretty_midi.PrettyMIDI):
            midi = prompt_midi
        else:
            midi = pretty_midi.PrettyMIDI()

        # ---------- 2️⃣ MIDI → tokens ----------
        prompt_tokens = self.model.midi_to_tokens(midi)

        # ---------- 3️⃣ 生成 tokens（核心） ----------
        gen_tokens = self.model.generate_tokens(
            prompt_tokens=prompt_tokens,
            bars=bars,
            constraints=constraints,
            temperature=temperature,
            top_p=top_p
        )

        # ---------- 4️⃣ tokens → events ----------
        events = self.model.tokens_to_events(gen_tokens)

        # ---------- 5️⃣ events → PrettyMIDI ----------
        midi = self._events_to_pretty_midi(events, midi)

        # ---------- 6️⃣ 保存 ----------
        if save:
            out_dir = Path("outputs")
            out_dir.mkdir(exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = out_dir / f"generated_{bars}bars_{ts}.mid"
            midi.write(str(out_path))
            print(f"🎵 Generated MIDI saved to: {out_path}")

        return midi

    # =========================================================
    # 🎹 events → PrettyMIDI
    # =========================================================
    def _events_to_pretty_midi(self, events, midi):
        piano = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program("Acoustic Grand Piano"))

        for e in events:
            note = pretty_midi.Note(
                velocity=e.get("velocity", 90),
                pitch=e["pitch"],
                start=e["start"],
                end=e["end"]
            )
            piano.notes.append(note)

        midi.instruments.append(piano)
        return midi
