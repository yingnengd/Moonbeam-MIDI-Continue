import torch

class MoonbeamWrapper:
    def __init__(self, ckpt_path,model, device="mps"):
        Self.ckpt_path=ckpt_path
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def generate_tokens(self, prompt_tokens, max_len=2046, temperature=1.0):
        tokens = prompt_tokens.clone().to(self.device)

        for _ in range(max_len):
            logits = self.model(tokens.unsqueeze(0))  # [1, T, V]
            next_logits = logits[0, -1] / temperature
            next_token = torch.multinomial(
                torch.softmax(next_logits, dim=-1), 1
            )
            tokens = torch.cat([tokens, next_token], dim=0)

        return tokens
