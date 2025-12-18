class HookMemory:
    def __init__(self):
        self.motif = None

    def store(self, notes):
        if self.motif is None:
            self.motif = notes[:8]

    def recall(self):
        return self.motif
