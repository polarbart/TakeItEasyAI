import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, input_size: int, n_atoms: int = 51, hidden_size: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LeakyReLU(),
            nn.Linear(hidden_size // 4, n_atoms)
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def expected_value(self, x: torch.Tensor):
        return self(x).mean(dim=-1)

    def save(self, file: str = 'net'):
        torch.save(self.state_dict(), file)

    @staticmethod
    def load(file: str = 'net'):
        d = torch.load(file)
        input_size = d['net.0.weight'].size(1)
        n_atoms = d['support'].numel()
        hidden_size = d['net.0.weight'].size(0)
        return Network(input_size, n_atoms, hidden_size)
