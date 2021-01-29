import numpy as np
import torch
pieces = np.array([[1, 2, 3],
                   [1, 2, 4],
                   [1, 2, 8],
                   [1, 6, 3],
                   [1, 6, 4],
                   [1, 6, 8],
                   [1, 7, 3],
                   [1, 7, 4],
                   [1, 7, 8],
                   [5, 2, 3],
                   [5, 2, 4],
                   [5, 2, 8],
                   [5, 6, 3],
                   [5, 6, 4],
                   [5, 6, 8],
                   [5, 7, 3],
                   [5, 7, 4],
                   [5, 7, 8],
                   [9, 2, 3],
                   [9, 2, 4],
                   [9, 2, 8],
                   [9, 6, 3],
                   [9, 6, 4],
                   [9, 6, 8],
                   [9, 7, 3],
                   [9, 7, 4],
                   [9, 7, 8]], dtype=np.int)

lines = [[[0, 1, 2], [3, 4, 5, 6], [7, 8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18]],
         [[0, 3, 7], [1, 4, 8, 12], [2, 5, 9, 13, 16], [6, 10, 14, 17], [11, 15, 18]],
         [[7, 12, 16], [3, 8, 13, 17], [0, 4, 9, 14, 18], [1, 5, 10, 15], [2, 6, 11]]]

lines_on_tile = [[0, 0, 2],
                 [0, 1, 3],
                 [0, 2, 4],
                 [1, 0, 1],
                 [1, 1, 2],
                 [1, 2, 3],
                 [1, 3, 4],
                 [2, 0, 0],
                 [2, 1, 1],
                 [2, 2, 2],
                 [2, 3, 3],
                 [2, 4, 4],
                 [3, 1, 0],
                 [3, 2, 1],
                 [3, 3, 2],
                 [3, 4, 3],
                 [4, 2, 0],
                 [4, 3, 1],
                 [4, 4, 2]]


class TakeItEasy:

    @property
    def empty_tiles(self):
        return np.argwhere(self.board == -1).squeeze(1).tolist()

    @property
    def remaining_pieces(self):
        return pieces[self.subset[self.step:]].tolist()

    @property
    def next_piece(self):
        if self.step >= 19:
            return None
        return pieces[self.subset[self.step]].tolist()

    def __init__(self, seed=None):
        np.random.seed(seed)
        self.board = -np.ones(19, dtype=np.int)
        self.subset = np.arange(27, dtype=np.int)
        self.last_positions = np.zeros(19, dtype=np.int)
        self.step = 0

    def set_next_piece(self, p):
        for i in range(27):
            if self.subset[i] == p:
                if i < self.step:
                    raise ValueError('piece was already played')
                self.swap_current_piece_with(i)
                break
        else:
            raise ValueError('invalid piece')

    def swap_current_piece_with(self, swp):
        self.subset[self.step], self.subset[swp] = self.subset[swp], self.subset[self.step]

    def reset(self):
        np.random.shuffle(self.subset)
        self.board[:] = -1
        self.last_positions[:] = -1
        self.step = 0

    def get_piece_at(self, i):
        if self.board[i] == -1:
            return None
        return pieces[self.board[i]].tolist()

    def compute_score(self):
        score = 0
        for i in range(3):
            for l in lines[i]:
                if np.all(self.board[l] != -1) and np.all(pieces[self.board[l]][:, i] == pieces[self.board[l[0]]][i]):
                    score += len(l) * pieces[self.board[l[0]]][i]
        return int(score)

    def _score_change(self, pos):
        score_change = 0
        for i, li in enumerate(lines_on_tile[pos]):
            l = lines[i][li]
            if np.all(self.board[l] != -1) and np.all(pieces[self.board[l]][:, i] == pieces[self.board[l[0]]][i]):
                score_change += len(l) * pieces[self.board[l[0]]][i]
        return int(score_change)

    def place(self, pos):
        if self.board[pos] != -1:
            raise RuntimeError('cant place hexagon at this pos')
        if self.step >= 19:
            raise RuntimeError('game has finished')
        self.last_positions[self.step] = pos
        self.board[pos] = self.subset[self.step]
        self.step += 1
        return self._score_change(pos)

    def undo(self):
        if self.step == 0:
            raise RuntimeError('nothing to undo')
        self.step -= 1
        self.board[self.last_positions[self.step]] = -1

    def encode(self):

        state = np.zeros((19, 3, 3), dtype=np.bool)

        for i in range(19):
            if self.board[i] != -1:
                h = pieces[self.board[i]]
                state[i][0][(h[0] - 1) // 4] = True
                state[i][1][0 if h[1] == 2 else (h[1] - 5)] = True
                state[i][2][2 if h[2] == 8 else (h[2] - 3)] = True

        return state.flatten()

    def __getstate__(self):
        return self.board, self.subset, self.step, self.last_positions

    def __setstate__(self, state):
        self.board, self.subset, self._step, self.last_positions = state
