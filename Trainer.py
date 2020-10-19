import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import torch.nn as nn
from torch.nn.functional import softmax, log_softmax, leaky_relu
import pickle
from tqdm import tqdm
from typing import Tuple, List

try:
    from TakeItEasyC import TakeItEasy
except ImportError:
    print('I do not use Cython')
    from TakeItEasy import TakeItEasy


class Buffer(Dataset):
    def __init__(self, size: int, state_size: int, n_atoms: int):
        self.s = torch.empty((size, state_size), dtype=torch.float)
        self.td = torch.empty((size, n_atoms), dtype=torch.float)
        self.index = 0

    def insert(self, s: torch.Tensor, td: torch.Tensor):
        if self.index >= len(self):
            raise RuntimeError('Buffer is full')
        self.s[self.index] = s
        self.td[self.index] = td
        self.index += 1
    
    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.s[i], self.td[i]

    def __len__(self) -> int:
        return self.s.shape[0]


class Network(nn.Module):

    def __init__(self, input_size, v_min, v_max, n_atoms=51, hidden_size=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_size // 2, n_atoms)
        )
        self.support = nn.Parameter(torch.linspace(v_min, v_max, n_atoms), requires_grad=False)

    def forward(self, x, log=False):
        y = self.net(x)
        return log_softmax(y, dim=-1) if log else softmax(y, dim=-1)

    def expected_value(self, x):
        return (self(x) * self.support).sum(dim=-1)

    def save(self, file='net'):
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file='net'):
        with open(file, 'rb') as f:
            return pickle.load(f)


class DenseNetwork(nn.Module):

    def __init__(self, input_size, v_min, v_max, n_atoms=51, hidden_size=512):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, n_atoms)
        self.support = nn.Parameter(torch.linspace(v_min, v_max, n_atoms), requires_grad=False)

    def forward(self, x, log=False):
        a = self.l1(x - .5)
        b = self.l2(leaky_relu(a))
        c = self.l3(leaky_relu(a + b))
        d = self.l4(leaky_relu(a + b + c))
        return log_softmax(d, dim=-1) if log else softmax(d, dim=-1)

    def expected_value(self, x):
        return (self(x) * self.support).sum(dim=-1)

    def save(self, file='net'):
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file='net'):
        with open(file, 'rb') as f:
            return pickle.load(f)


class Trainer:
    def __init__(
            self,
            lr: float = 1e-4,
            lr_decay: float = .97,
            n_games: int = 512,
            n_atoms: int = 51,
            n_epochs: int = 16,
            hidden_neurons: int = 512,
            epsilon: float = .2,
            epsilon_decay: float = .97,
            batch_size: int = 256,
    ):
        self.game = TakeItEasy()
        self.n_games = n_games
        self.n_atoms = n_atoms
        self.n_epochs = n_epochs
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        self.state_size = 3*5*5+19*3*3+27
        self.v_min, self.v_max = 0, 307
        self.iteration = 1

        self.net = DenseNetwork(self.state_size, self.v_min, self.v_max, self.n_atoms, hidden_neurons)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, lr_decay)

        self._support = torch.linspace(self.v_min, self.v_max, self.n_atoms, dtype=torch.float)
        self._td = torch.empty((self.n_atoms,), dtype=torch.float)
        self._delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)

        self.validation_scores = []
        self.train_scores = []

    def create_dataset(self) -> Tuple[Buffer, List[int]]:
        buffer = Buffer(18 * self.n_games, self.state_size, self.n_atoms)
        scores = []

        for _ in tqdm(range(self.n_games)):

            # play one game
            self.game.reset()
            empty_tiles = list(range(19))
            for step in range(19):

                # we do not need the explicit value distribution of the first state
                remaining_pieces = (27 - step) if step > 0 else 1
                # remaining_pieces = (27 - step) if step > 0 and step >= 19 - self.iteration else 1
                # remaining_pieces = 1
                states1 = torch.empty((remaining_pieces, len(empty_tiles), self.state_size), dtype=torch.float)
                rewards = torch.empty((remaining_pieces, len(empty_tiles)), dtype=torch.float)

                # iterate through all pieces that could be next
                for i in range(remaining_pieces):
                    self.game.swap_current_piece_with(i + step)

                    # iterate though all tiles that the piece could be placed on
                    for j, a in enumerate(empty_tiles):
                        rewards[i, j], _ = self.game.place(a)
                        states1[i, j] = self.game.encode()
                        self.game.undo()
                    self.game.swap_current_piece_with(i + step)

                # compute the value distribution of all next states
                if step < 18 and self.iteration > 1:
                    # only use the network if its output is reasonable
                    with torch.no_grad():
                        qd1 = self.net(states1)  # q-distribution_(t+1)
                else:
                    # otherwise assume that the agent will get no future reward
                    # important! only works because self._support[0] == 0
                    qd1 = torch.zeros((remaining_pieces, len(empty_tiles), self.n_atoms), dtype=torch.float)
                    qd1[:, :, 0] = 1

                expected_future_rewards = rewards + (qd1 * self._support).sum(dim=2)
                argmax = expected_future_rewards.argmax(dim=1)

                # we do not need the value distribution of the first state
                if step > 0:  # step > 0
                    # compute the target distribution
                    r = rewards.gather(1, argmax.unsqueeze(1)).squeeze(1)
                    d1 = qd1.gather(1, argmax.view(-1, 1, 1).expand(remaining_pieces, 1, self.n_atoms)).squeeze(1)

                    Tz = (r.unsqueeze(1) + self._support).clamp_(self.v_min, self.v_max)
                    b = (Tz - self.v_min) / self._delta_z
                    l = b.floor().long()
                    u = b.ceil().long()
                    l[(u > 0) & (l == u)] -= 1
                    u[(l == u)] += 1

                    self._td.zero_()
                    self._td.index_add_(0, l.view(-1), (d1 * (u - b)).view(-1))
                    self._td.index_add_(0, u.view(-1), (d1 * (b - l)).view(-1))
                    self._td /= remaining_pieces

                    state = self.game.encode()  # insert state and target distribution into buffer
                    buffer.insert(state, self._td)

                # apply action
                if random.random() < self.epsilon:
                    action = random.choice(empty_tiles)
                else:
                    action = empty_tiles[int(argmax[0])]
                self.game.place(action)
                empty_tiles.remove(action)

            scores.append(self.game.compute_score())
        return buffer, scores

    def validate(self, n_games: int):
        rewards = []
        for _ in range(n_games):
            self.game.reset()
            empty_tiles = set(range(19))
            for step in range(18):
                tmp = []
                for a in empty_tiles:
                    r, _ = self.game.place(a)
                    tmp.append((a, self.game.encode(), r))
                    self.game.undo()
                with torch.no_grad():
                    q = torch.tensor([s[2] for s in tmp], dtype=torch.float) + self.net.expected_value(torch.cat([s[1] for s in tmp], dim=0).float())
                action = tmp[q.argmax()][0]
                empty_tiles.remove(action)
                self.game.place(action)
                rewards.append(tmp[q.argmax()][2])
            rewards.append(self.game.place(empty_tiles.pop())[0])
        return np.mean(rewards) * 19

    def train(self, validate_every: int = 1, save_name='trainer'):
        while True:
            dataset, scores = self.create_dataset()
            dataloader = DataLoader(dataset, self.batch_size, True, drop_last=True)
            self.train_scores.append(np.mean(scores))
            print(f'{self.iteration} {self.train_scores[-1]:.2f}', end=' ')
            for _ in range(self.n_epochs):
                losses = []
                for s, td in dataloader:
                    self.optimizer.zero_grad()
                    ld = self.net(s, log=True)
                    loss = -(ld * td).sum(dim=1).mean()
                    loss.backward()
                    self.optimizer.step()
                    losses.append(float(loss))
                print(f'{np.mean(losses):.2f}', end=' ')
            print()
            self.lr_scheduler.step()
            self.epsilon *= self.epsilon_decay
            self.iteration += 1
            if self.iteration % validate_every == 0:
                self.validation_scores.append((self.iteration, self.validate(5000)))
                print(self.iteration, 'validation: ', self.validation_scores[-1][1])
            self.save(save_name)

    def save(self, file: str = 'trainer'):
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file: str = 'trainer'):
        with open(file, 'rb') as f:
            return pickle.load(f)


if __name__ == '__main__':
    load = False
    save_name = 'trainer_test'
    # print(Trainer.load(save_name).validation_scores); exit()
    if load:
        t = Trainer.load(save_name)
        # t.lr_scheduler.last_epoch = 5
    else:
        # /sgd ausprobieren
        # /erwartungswert nicht explizit
        # 2048
        # /besseres netzwerk
        # erwartungswert überprüfen
        # parameter explizit auflisten
        # /densenet ohne x - .5
        # von hinten nach vorne trainieren
        # lr_decay für jeden training abschnitt, nicht global
        # batch normalization
        # Fast as Adam & as Good as SGD
        # amsgrad
        t = Trainer(lr=3e-4, epsilon=.5, hidden_neurons=512, n_games=4096, n_epochs=16, epsilon_decay=.98, lr_decay=.99)  # 1024
    t.train(validate_every=5, save_name=save_name)
