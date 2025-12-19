import torch
from pathlib import Path
import pretty_midi
from datetime import datetime

# ==============================
# 🔹 工具函数
# ==============================
def midi_to_prompt_tokens(midi: pretty_midi.PrettyMIDI):
    """
    将 PrettyMIDI 转换为模型输入 token
    这里需要根据 Moonbeam 模型要求实现
    """
    # 示例占位：每个 note 转为字典
    tokens = []
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            tokens.append({
                "pitch": note.pitch,
                "start": note.start,
                "end": note.end,
                "velocity": note.velocity
            })
    return tokens

def tokens_to_pretty_midi(tokens, instrument=0):
    """
    将 token 转为 PrettyMIDI
    """
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=instrument)
    for t in tokens:
        note = pretty_midi.Note(
            pitch=t.get("pitch", 60),
            start=t.get("start", 0.0),
            end=t.get("end", 0.5),
            velocity=t.get("velocity", 100)
        )
        inst.notes.append(note)
    midi.instruments.append(inst)
    return midi

# ==============================
# 🔹 MusicLLaMA / Moonbeam 模型封装
# ==============================
class MusicLLaMA:
    def __init__(self, ckpt_path, device="mps"):
        self.device = torch.device(device if torch.backends.mps.is_built() or device != "mps" else "cpu")
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        # 🔹 加载权重
        self.model_state = torch.load(str(ckpt_path), map_location=self.device)
        print(f"✅ Loaded checkpoint: {ckpt_path}, device: {self.device}")

        # TODO: 初始化真实 Moonbeam 模型
        # self.net = MoonbeamModelClass().to(self.device)
        # self.net.load_state_dict(self.model_state)

    def generate(self, prompt_midi=None, bars=8, constraints=None, temperature=1.0, top_p=0.9):
        """
        生成 MIDI，返回 PrettyMIDI 对象
        """
        if prompt_midi is None:
            prompt_midi_obj = pretty_midi.PrettyMIDI()
        elif isinstance(prompt_midi, str):
            prompt_midi_obj = pretty_midi.PrettyMIDI(prompt_midi)
        elif isinstance(prompt_midi, pretty_midi.PrettyMIDI):
            prompt_midi_obj = prompt_midi
        else:
            raise TypeError(f"prompt_midi must be str, PrettyMIDI, or None, got {type(prompt_midi)}")

        # 转为模型 token
        prompt_tokens = midi_to_prompt_tokens(prompt_midi_obj)

        # 🔹 调用真实模型生成 token
        tokens = self._model_generate_tokens(prompt_tokens, bars, constraints, temperature, top_p)

        # token → PrettyMIDI
        midi_out = tokens_to_pretty_midi(tokens)

        # 保存 MIDI
        out_dir = Path("outputs")
        out_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"generated_{bars}bars_{timestamp}.mid"
        midi_out.write(str(out_path))
        print(f"✅ Generated MIDI saved to {out_path}")

        return midi_out

    def _model_generate_tokens(self, prompt_tokens, bars, constraints, temperature, top_p):
        """
        核心生成函数：将 prompt token 输入 Moonbeam，生成连续小节
        返回 token list [{"pitch", "start", "end", "velocity"}]
        """
        # TODO: 替换为真实模型推理逻辑
        # 示例占位逻辑（生成小节序列）
        beats_per_bar = 4
        time_per_beat = 0.5
        tonic = constraints.get("tonic", 60) if constraints else 60

        tokens = []
        for i in range(bars * beats_per_bar):
            pitch = tonic + (i % 12)  # 这里换成模型输出
            start = i * time_per_beat
            end = start + time_per_beat
            velocity = 100
            tokens.append({"pitch": pitch, "start": start, "end": end, "velocity": velocity})

        return tokens

# ==============================
# 🔹 运行示例
# ==============================
if __name__ == "__main__":
    model = MusicLLaMA("moonbeam_839M.pt", device="mps")

    # 可以传入 prompt MIDI 或 None
    midi = model.generate(
        prompt_midi=None,
        bars=8,
        constraints={"tonic": 60, "mode": "major"}
    )
