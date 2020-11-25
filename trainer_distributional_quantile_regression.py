import torch
import numpy as np
from Network import Network
from torch.nn.functional import smooth_l1_loss
import pickle
from tqdm import tqdm
import sys

try:
    from cpp.build.Release.TakeItEasyC import BatchedTakeItEasy, TakeItEasy
except ImportError:
    try:
        from cpp.build.TakeItEasyC import BatchedTakeItEasy, TakeItEasy
    except ImportError:
        raise ImportError("You need to compile the TakeItEasy C++ implementation as described in the readme")


class Buffer:

    """
    Holds the data which is later used for training
    """

    def __init__(self, size: int, state_size: int, n_atoms: int, device: str, data_device: str, n_dist: int = 26):
        self.s = torch.empty((size, state_size), dtype=torch.int8, device=data_device)
        self.td = -torch.ones((size, n_dist, n_atoms), dtype=torch.half, device=data_device)
        self.n_samples = torch.empty(size, dtype=torch.int8, device=data_device)
        self.device = device
        self.data_device = data_device
        self.index = 0

    def insert(self, s: torch.Tensor, td: torch.Tensor):
        batch_size = s.shape[0]
        if self.index + batch_size > self.s.shape[0]:
            raise RuntimeError('Buffer is full')
        self.s[self.index:self.index+batch_size] = s.to(self.data_device)
        self.td[self.index:self.index+batch_size, :td.size(1)] = td.to(self.data_device)
        self.n_samples[self.index:self.index+batch_size] = td.size(1)
        self.index += batch_size

    def __getitem__(self, idx: torch.Tensor):
        return self.s[idx].to(self.device).float(), \
               self.td[idx].to(self.device).float(), \
               self.n_samples[idx].to(self.device).long()

    def __len__(self):
        return self.s.size(0)


class CustomDataLoader:

    """
    A custom dataloader which is significantly faster compared to the PyTorch dataloader if the dataset is already on the gpu
    """

    def __init__(self, dataset: Buffer, batch_size: int, device: str, data_device: str):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_batches = len(dataset) // batch_size
        self.device = device
        self.data_device = data_device
        self.batches = None
        self.idx = None

    def __iter__(self):
        self.batches = torch.randperm(len(self.dataset), device=self.data_device)[:self.num_batches * self.batch_size].view((self.num_batches, self.batch_size))
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= self.num_batches:
            raise StopIteration
        ret = self.dataset[self.batches[self.idx]]
        self.idx += 1
        return ret

    def __len__(self):
        return self.num_batches


class Trainer:
    def __init__(
            self,
            lr: float = 1e-4,
            lr_decay: float = .97,
            n_games: int = 4096,
            n_validation_games: int = 4096,
            n_atoms: int = 51,
            n_epochs: int = 16,
            n_iterations: int = None,
            hidden_neurons: int = 512,
            epsilon: float = .2,
            epsilon_decay: float = .97,
            batch_size: int = 256,
            batch_size_games: int = 1024,
            device: str = 'cuda',
            data_device: str = 'cuda'
    ):
        assert n_games % batch_size_games == 0 and n_validation_games % batch_size_games == 0, \
            'n_games must and n_validation_games be a multiple of batch_size_games'
        self.lr = lr
        self.lr_decay = lr_decay
        self.n_games = n_games
        self.n_validation_games = n_validation_games
        self.batch_size_games = batch_size_games
        self.n_atoms = n_atoms
        self.n_epochs = n_epochs
        self.n_iterations = n_iterations
        self.hidden_neurons = hidden_neurons
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.device = device
        self.data_device = data_device

        self.games = BatchedTakeItEasy(batch_size_games)

        self.state_size = 19 * 3 * 3
        self.iteration = 1

        self.net = Network(self.state_size, self.n_atoms, self.hidden_neurons).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), self.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, self.lr_decay)

        self.tau = ((2 * torch.arange(n_atoms, device=device, dtype=torch.float) + 1) / (2 * n_atoms)).unsqueeze(1)

        self.validation_scores = []
        self.train_scores = []

    def _quantile_regression_loss(self, qd: torch.Tensor, tqd: torch.Tensor, n_samples: torch.Tensor):  # batch_size, n_atoms; batch_size, 26, n_atoms
        tqd = tqd.view(tqd.size(0), 1, -1)  # batch_size, 1, (26*n_atoms)
        qd = qd.unsqueeze(2)  # batch_size, n_atoms, 1
        mask = (tqd != -1)  # batch_size, 1, (26*n_atoms)
        weight = torch.abs((self.tau - (tqd < qd.detach()).float())) / n_samples.view(-1, 1, 1)
        qd, tqd = torch.broadcast_tensors(qd, tqd)  # otherwise 'smooth_l1_loss' will throw a warning which is annoying
        return (weight * mask * smooth_l1_loss(qd, tqd, reduction='none')).sum() / self.batch_size

    @torch.no_grad()
    def create_dataset(self):

        def to_torch_tensor(x):
            if x is not None:
                return torch.from_numpy(x).to(self.device)

        buffer = Buffer(18 * self.n_games, self.state_size, self.n_atoms, self.device, self.data_device)
        scores = []

        self.net.eval()
        for _ in tqdm(range(self.n_games // self.batch_size_games), desc=f'Creating Dataset {self.iteration:<3}', file=sys.stdout):

            self.games.reset()

            for step in range(19):
                '''
                  states.shape = batch_size_games x state_size
              
                  if step > 0:
                    compute_encodings returns all possible next states (iterating over empty tiles and remaining pieces)
                    states1.shape = batch_size_games x (27 - step) x (19 - step) x state_size
                    rewards.shape = batch_size_games x (27 - step) x (19 - step)
                  elif step == 0:
                    compute_encodings returns all possible next states (just iterating over empty tiles)
                    states1.shape = batch_size_games x 1 x (19 - step) x state_size
                    rewards.shape = batch_size_games x 1 x (19 - step)
              
                  empty_tiles.shape = batch_size_games x (19 - step)
              
                  if step == 0 then states is None
                  if step == 18 then states1 is None
                '''
                states, states1, rewards, empty_tiles = self.games.compute_encodings(step > 0)
                states, states1, rewards = to_torch_tensor(states), to_torch_tensor(states1), to_torch_tensor(rewards)

                n_games, n_remaining_pieces, n_empty_tiles = rewards.shape

                # compute the value distribution of all next states
                if step < 18 and self.iteration > 1:
                    # only use the network if its output is reasonable
                    qd1 = self.net(states1.float())  # q-distribution_(t+1), shape: n_games, n_remaining_pieces, n_empty_tiles, self.n_atoms
                else:
                    # otherwise assume that the agent will get no future reward
                    qd1 = torch.zeros((n_games, n_remaining_pieces, n_empty_tiles, self.n_atoms), dtype=torch.float, device=self.device)

                expected_future_rewards = rewards + qd1.mean(dim=3)  # n_games, n_remaining_pieces, n_empty_tiles
                best_action = expected_future_rewards.argmax(dim=2)  # n_games, n_remaining_pieces

                # we do not need the value distribution of the initial state
                if step > 0:  # step > 0
                    # compute the target distribution
                    r = rewards.gather(2, best_action.unsqueeze(2))  # n_games, n_remaining_pieces, 1
                    d1 = qd1.gather(2, best_action.view(n_games, n_remaining_pieces, 1, 1).expand(n_games, n_remaining_pieces, 1, self.n_atoms)).squeeze(2)  # n_games, n_remaining_pieces, self.n_atoms

                    buffer.insert(states, r + d1)

                actions = np.where(
                    np.random.ranf(n_games) < self.epsilon,
                    np.random.randint(0, n_empty_tiles, size=(n_games,), dtype=np.int8),
                    best_action[:, 0].cpu().numpy().astype(np.int8)
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

    def train(self, validate_every: int = 1, save_name: str = 'trainer'):

        while self.n_iterations is None or self.iteration <= self.n_iterations:

            # create dataset
            dataset, scores = self.create_dataset()
            dataloader = CustomDataLoader(dataset, self.batch_size, self.device, self.data_device)
            self.train_scores.append(np.mean(scores))

            # train
            self.net.train()
            for _ in tqdm(range(self.n_epochs), desc=f'Training {self.iteration:<11}', file=sys.stdout):
                losses = []
                for s, tqds, n_samples in dataloader:
                    self.optimizer.zero_grad()
                    qd = self.net(s)
                    loss = self._quantile_regression_loss(qd, tqds, n_samples)
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

    def __getstate__(self):
        state = self.__dict__.copy()
        state['net'] = self.net.state_dict()
        state['optimizer'] = self.optimizer.state_dict()
        state['lr_scheduler'] = self.lr_scheduler.state_dict()
        return state

    def __setstate__(self, state):

        net = Network(state['state_size'], state['n_atoms'], state['hidden_neurons']).to(state['device'])
        optimizer = torch.optim.Adam(net.parameters(), state['lr'])
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, state['lr_decay'])

        net.load_state_dict(state['net'])
        optimizer.load_state_dict(state['optimizer'])
        lr_scheduler.load_state_dict(state['lr_scheduler'])

        state['net'] = net
        state['optimizer'] = optimizer
        state['lr_scheduler'] = lr_scheduler

        self.__dict__ = state

    def save(self, file: str = 'trainer'):
        with open(file, 'wb') as f:
            pickle.dump(self, f)
        torch.save(self.net.state_dict(), file + '_net')

    @staticmethod
    def load(file: str = 'trainer'):
        with open(file, 'rb') as f:
            return pickle.load(f)


if __name__ == '__main__':
    load = False
    save_name = 'trainer_qr'
    if load:
        t = Trainer.load(save_name)
    else:
        t = Trainer(
            lr=3e-4,
            epsilon=.5,
            batch_size=128,
            hidden_neurons=2048,
            n_games=16384,
            batch_size_games=1024,
            n_validation_games=16384,
            n_epochs=8,
            epsilon_decay=.85,
            lr_decay=.965,
            n_atoms=100,
            n_iterations=150,
            device='cuda',
            data_device='cuda'  # the dataset is saved on the gpu, set to 'cpu' if not enough VRAM
        )
    t.train(validate_every=5, save_name=save_name)
