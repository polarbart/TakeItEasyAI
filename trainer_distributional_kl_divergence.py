import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import torch.nn as nn
from torch.nn.functional import softmax, log_softmax, leaky_relu
import pickle
from tqdm import tqdm
from typing import Tuple, List
from time import time
import sys

try:
    from cpp.build.Release.TakeItEasyC import BatchedTakeItEasy
except ImportError:
    raise RuntimeError("you need to compile the TakeItEasy C extension")


class CustomDataLoader:
    def __init__(self, dataset, batch_size, device):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_batches = len(dataset) // batch_size
        self.device = device
        self.batches = None
        self.idx = None

    def __iter__(self):
        self.batches = torch.randperm(self.num_batches * self.batch_size, device=self.device).view((self.num_batches, self.batch_size))
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= self.num_batches:
            raise StopIteration
        ret = self.dataset[self.batches[self.idx]]
        # ret = self.dataset.s[self.batches[self.idx]], self.dataset.td[self.batches[self.idx]]
        self.idx += 1
        return ret

    def __len__(self):
        return self.num_batches


class Buffer(Dataset):
    def __init__(self, size, state_size, n_atoms, device):
        self.s = torch.empty((size, state_size), dtype=torch.float, device=device)
        self.td = torch.empty((size, n_atoms), dtype=torch.float, device=device)
        self.index = 0

    def insert(self, s: torch.Tensor, td: torch.Tensor):
        batch_size = s.shape[0]
        if self.index + batch_size > self.s.shape[0]:
            raise RuntimeError('Buffer is full')
        self.s[self.index:self.index+batch_size] = s
        self.td[self.index:self.index+batch_size] = td
        self.index += batch_size
    
    def __getitem__(self, idx):
        return self.s[idx], self.td[idx]

    def __len__(self):
        return self.index


class Network(nn.Module):

    def __init__(self, input_size, v_min, v_max, n_atoms=51, hidden_size=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, n_atoms)
        )
        self.support = nn.Parameter(torch.linspace(v_min, v_max, n_atoms), requires_grad=False)

    def forward(self, x, log=False):
        y = self.net(x)
        return log_softmax(y, dim=-1) if log else softmax(y, dim=-1)

    def expected_value(self, x):
        return (self(x) * self.support).sum(dim=-1)

    def save(self, file='net'):
        torch.save(self.state_dict(), file)

    @staticmethod
    def load(file='net'):
        d = torch.load(file)
        input_size = d['net.0.weight'].size(1)
        v_min = d['support'].min()
        v_max = d['support'].max()
        n_atoms = d['support'].numel()
        hidden_size = d['net.0.weight'].size(0)
        return Network(input_size, v_min, v_max, n_atoms, hidden_size)


class Trainer:
    def __init__(
            self,
            lr: float = 1e-4,
            lr_decay: float = .97,
            n_games: int = 512,
            n_validation_games: int = 4096,
            n_atoms: int = 51,
            n_epochs: int = 16,
            n_iterations: int = None,
            hidden_neurons: int = 512,
            epsilon: float = .2,
            epsilon_decay: float = .97,
            batch_size: int = 256,
            batch_size_games: int = 1024,
            device: str = 'cuda'
    ):
        assert n_games % batch_size_games == 0 and n_validation_games % batch_size_games == 0, \
            'n_games must and n_validation_games be a multiple of batch_size_games'
        self.games = BatchedTakeItEasy(batch_size_games)
        self.n_games = n_games
        self.n_validation_games = n_validation_games
        self.batch_size_games = batch_size_games
        self.n_atoms = n_atoms
        self.n_epochs = n_epochs
        self.n_iterations = n_iterations
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.device = device

        # self.state_size = 3*5*5+19*3*3+27
        self.state_size = 19 * 3 * 3
        self.v_min, self.v_max = 0, 307
        self.iteration = 1

        self.net = Network(self.state_size, self.v_min, self.v_max, self.n_atoms, hidden_neurons).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, lr_decay)

        self._support = torch.linspace(self.v_min, self.v_max, self.n_atoms, dtype=torch.float, device=device)
        self._td = torch.empty((batch_size_games, self.n_atoms), dtype=torch.float, device=device)
        self._delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)
        self._offset = (torch.arange(0, batch_size_games, device=device) * self.n_atoms).view(batch_size_games, 1, 1)

        self.validation_scores = []
        self.train_scores = []

    @torch.no_grad()
    def create_dataset(self):

        def to_torch_tensor(x):
            if x is not None:
                return torch.from_numpy(x).to(self.device).float()

        buffer = Buffer(18 * self.n_games, self.state_size, self.n_atoms, self.device)
        scores = []

        self.net.eval()
        for _ in tqdm(range(self.n_games // self.batch_size_games), desc=f'Creating Dataset {self.iteration:<3}', file=sys.stdout):

            self.games.reset()

            for step in range(19):
                states, states1, rewards, empty_tiles = self.games.compute_encodings(step > 0)

                states, states1, rewards = to_torch_tensor(states), to_torch_tensor(states1), to_torch_tensor(rewards)

                n_remaining_pieces = (27 - step) if step > 0 else 1
                n_empty_tiles = 19 - step
                n_games = states1.shape[0]

                # compute the value distribution of all next states
                if step < 18 and self.iteration > 1:
                    # only use the network if its output is reasonable
                    qd1 = self.net(states1)  # q-distribution_(t+1), shape: n_games, n_remaining_pieces, n_empty_tiles, self.n_atoms
                else:
                    # otherwise assume that the agent will get no future reward
                    # important! only works because self._support[0] == 0
                    qd1 = torch.zeros((n_games, n_remaining_pieces, n_empty_tiles, self.n_atoms), dtype=torch.float, device=self.device)
                    qd1[:, :, :, 0] = 1

                expected_future_rewards = rewards + (qd1 * self._support).sum(dim=3)  # n_games, n_remaining_pieces, n_empty_tiles
                argmax = expected_future_rewards.argmax(dim=2)  # n_games, n_remaining_pieces

                # we do not need the value distribution of the first state
                if step > 0:  # step > 0
                    # compute the target distribution
                    r = rewards.gather(2, argmax.unsqueeze(2)).squeeze(2)  # n_games, n_remaining_pieces
                    d1 = qd1.gather(2, argmax.view(n_games, n_remaining_pieces, 1, 1).expand(n_games, n_remaining_pieces, 1, self.n_atoms)).squeeze(2)  # n_games, n_remaining_pieces, self.n_atoms

                    Tz = (r.unsqueeze(2) + self._support).clamp_(self.v_min, self.v_max)  # n_games, n_remaining_pieces, self.n_atoms
                    b = (Tz - self.v_min) / self._delta_z
                    l = b.floor().long()  # n_games, n_remaining_pieces, self.n_atoms
                    u = b.ceil().long()  # n_games, n_remaining_pieces, self.n_atoms
                    l.clamp_(0, self.n_atoms - 1)  # rounding errors may occur
                    u.clamp_(0, self.n_atoms - 1)  # rounding errors may occur
                    l[(u > 0) & (l == u)] -= 1
                    u[(l == u)] += 1

                    self._td.zero_()
                    self._td.view(-1).index_add_(0, (l + self._offset).view(-1), (d1 * (u - b)).view(-1))
                    self._td.view(-1).index_add_(0, (u + self._offset).view(-1), (d1 * (b - l)).view(-1))
                    self._td /= n_remaining_pieces

                    buffer.insert(states, self._td)

                actions = np.where(
                    np.random.ranf(n_games) < self.epsilon,
                    np.random.randint(0, n_empty_tiles, size=(n_games,), dtype=np.int8),
                    argmax[:, 0].cpu().numpy().astype(np.int8)
                )
                actions = np.take_along_axis(empty_tiles, actions.reshape(n_games, 1), axis=1).reshape(n_games)
                self.games.place(actions)

            scores += self.games.compute_scores().tolist()

        return buffer, scores

    @torch.no_grad()
    def validate(self):

        def to_torch_tensor(x):
            if x is not None:
                return torch.from_numpy(x).to(self.device).float()

        rewards = []

        self.net.eval()
        for _ in range(self.n_validation_games // self.batch_size_games):

            self.games.reset()

            for step in range(18):
                _, s1, r, empty_tiles = self.games.compute_encodings(False)
                s1, r = to_torch_tensor(s1), to_torch_tensor(r)
                e = r.squeeze(1) + self.net.expected_value(s1.squeeze(1))  # n_games, n_empty_tiles
                actions = np.take_along_axis(empty_tiles, e.argmax(dim=1, keepdim=True).cpu().numpy(), axis=1).squeeze(1)
                self.games.place(actions)

            *_, empty_tiles = self.games.compute_encodings(False)
            self.games.place(empty_tiles.squeeze(1))

            rewards += self.games.compute_scores().tolist()

        return np.mean(rewards)

    def train(self, validate_every: int = 1, save_name='trainer'):

        while self.n_iterations is None or self.iteration <= self.n_iterations:

            # create dataset
            dataset, scores = self.create_dataset()
            dataloader = CustomDataLoader(dataset, self.batch_size, self.device)
            self.train_scores.append(np.mean(scores))

            # train
            self.net.train()
            for _ in tqdm(range(self.n_epochs), desc=f'Training {self.iteration:<11}', file=sys.stdout):
                losses = []
                for s, td in dataloader:
                    self.optimizer.zero_grad()
                    ld = self.net(s, log=True)
                    loss = -(ld * td).sum(dim=1).mean()
                    loss.backward()
                    self.optimizer.step()
                    losses.append(loss.item())

            self.lr_scheduler.step()
            self.epsilon *= self.epsilon_decay

            if self.iteration % validate_every == 0:
                self.validation_scores.append((self.iteration, self.validate()))
                print(f'Validation {self.iteration}: {self.validation_scores[-1][1]:.2f}')

            self.iteration += 1
            self.save(save_name)

    #def __getstate__(self):
    #    state = self.__dict__
    #    state['net'] = self.net.state_dict()
    #    state['optimizer'] = self.optimizer.state_dict()
    #    state['lr_scheduler'] = self.lr_scheduler.state_dict()
    #    return state

    #def __setstate__(self, state):
    #    state['net'] = self.net.state_dict()
    #    state['optimizer'] = self.optimizer.state_dict()
    #    state['lr_scheduler'] = self.lr_scheduler.state_dict()
    #    self.__dict__ = state

    def save(self, file: str = 'trainer'):
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file: str = 'trainer'):
        with open(file, 'rb') as f:
            return pickle.load(f)


if __name__ == '__main__':
    load = False
    save_name = 'trainer_batched_4'  # 'trainer_batched_2' 1024x1024
    if load:
        t = Trainer.load(save_name)
    else:
        t = Trainer(
            lr=3e-4,
            epsilon=.5,
            batch_size=128,
            hidden_neurons=512,
            n_games=16384,
            batch_size_games=512,
            n_validation_games=16384,
            n_epochs=8,
            epsilon_decay=.85,
            lr_decay=.965,
            n_atoms=101,
            n_iterations=150,
            device='cuda'
        )
    t.train(validate_every=5, save_name=save_name)
    for s in t.validation_scores:
        print(s)
