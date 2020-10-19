#distutils: language=c++
# cythonize -i TakeItEasyC.pyx
cimport cython
from cython.parallel import prange
from libc.stdlib cimport rand, srand
from libc.time cimport time
from libcpp cimport bool
from libcpp.vector cimport vector
import torch
import numpy as np
cimport numpy as np
from Trainer import Network

cdef const int[27][3] pieces = [[1, 2, 3],
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
                                [9, 7, 8]]

cdef const int[3][5][5] lines = [[[0, 1, 2, 0, 0], [3, 4, 5, 6, 0], [7, 8, 9, 10, 11], [12, 13, 14, 15, 0], [16, 17, 18, 0, 0]],
                                 [[0, 3, 7, 0, 0], [1, 4, 8, 12, 0], [2, 5, 9, 13, 16], [6, 10, 14, 17, 0], [11, 15, 18, 0, 0]],
                                 [[7, 12, 16, 0, 0], [3, 8, 13, 17, 0], [0, 4, 9, 14, 18], [1, 5, 10, 15, 0], [2, 6, 11, 0, 0]]]

cdef const int[19][3] lines_on_tile = [[0, 0, 2],
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

cdef const int state_size = 3*5*5 + 19*3*3 + 27

cdef bool all_equal(int array[], int array_size, int value):
    for i in range(array_size):
        if array[i] != value:
            return False
    return True


cdef class TakeItEasy:

    cdef int board_[19]
    cdef int subset_[27]
    cdef int last_positions_[19]
    cdef int step_

    @property
    def step(self):
        return self.step_

    @property
    def empty_tiles(self):
        ret = []
        for i in range(19):
            if self.board_[i] == -1:
                ret.append(i)
        return ret

    @property
    def remaining_pieces(self):
        ret = []
        for i in range(self.step_, 27):
            ret.append(pieces[self.subset_[i]])
        return ret

    @property
    def next_piece(self):
        if self.step_ >= 19:
            return None
        return pieces[self.subset_[self.step_]]

    def __cinit__(self, seed=None):
        cdef int i
        for i in range(19):
            self.board_[i] = -1
            self.last_positions_[i] = -1
        for i in range(27):
            self.subset_[i] = i
        if seed is None:
            srand(time(NULL))
        else:
            srand(seed)
        self.step_ = 0

    def set_next_piece(self, int p):
        for i in range(27):
            if self.subset_[i] == p:
                if i < self.step_:
                    raise ValueError('piece was already played')
                self.subset_[self.step_], self.subset_[i] = self.subset_[i], self.subset_[self.step_]
                break
        else:
            raise ValueError('invalid piece')

    cdef void swap_current_piece_with_(self, int swp):
        if swp < self.step_:
            raise ValueError('piece was already played')
        self.subset_[self.step_], self.subset_[swp] = self.subset_[swp], self.subset_[self.step_]

    def swap_current_piece_with(self, int swp):
        self.swap_current_piece_with_(swp)

    @cython.cdivision(True)
    cdef reset_(self):
        cdef int r, i
        for i in range(19):
            r = rand() % (27 - i)
            self.subset_[i], self.subset_[i + r] = self.subset_[i + r], self.subset_[i]
        for i in range(19):
            self.board_[i] = -1
            self.last_positions_[i] = -1
        self.step_ = 0

    def reset(self):
        self.reset_()

    def get_piece_at(self, int i):
        if self.board_[i] == -1:
            return None
        return pieces[self.board_[i]]

    def compute_score(self):
        cdef int score = 0, reference, number_of_tiles, j, p
        for i in range(3):
            for j in range(5):
                number_of_tiles = 5 - abs(2 - j)
                p = self.board_[lines[i][j][0]]
                if p != -1:
                    reference = pieces[p][i]
                    for k in range(1, number_of_tiles):
                        p = self.board_[lines[i][j][k]]
                        if p == -1 or pieces[p][i] != reference:
                            break
                    else:
                        score += number_of_tiles * reference
        return score

    cdef int _score_change(self, int pos):
        cdef int score_change = 0, number_of_tiles, p, i, l, j
        cdef int[3] reference = pieces[self.board_[pos]]
        for i in range(3):
            l = lines_on_tile[pos][i]
            number_of_tiles = 5 - abs(2 - l)
            for j in range(number_of_tiles):
                p = self.board_[lines[i][l][j]]
                if p == -1 or pieces[p][i] != reference[i]:
                    break
            else:
                score_change += number_of_tiles * reference[i]
        return score_change

    cdef int place_(self, int pos):
        if self.board_[pos] != -1:
            raise RuntimeError('there is already a piece on this tile')
        if self.step_ >= 19:
            raise RuntimeError('game has finished')

        self.last_positions_[self.step_] = pos
        self.board_[pos] = self.subset_[self.step_]
        self.step_ += 1
        return self._score_change(pos)

    def place(self, int pos):
        return self.place_(pos), self.step_ == 19

    def undo(self):
        if self.step_ == 0:
            raise RuntimeError('nothing to undo')
        self.step_ -= 1
        self.board_[self.last_positions_[self.step_]] = -1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef encode_(self, np.float32_t [::1] buf):
        assert len(buf) == state_size
        cdef np.float32_t [:, :, ::1] state1_view = &buf[0]
        cdef np.float32_t [:, :, ::1] state2_view = &buf[3*5*5]
        cdef np.float32_t [::1] state3_view = &buf[3*5*5+19*3*3]

        cdef int j, number_of_tiles, t, p
        cdef bool complete
        for i in range(3):
            for j in range(5):
                t = -1
                complete = True
                number_of_tiles = 5 - abs(2-j)
                for k in range(number_of_tiles):
                    p = self.board_[lines[i][j][k]]
                    if p != -1:
                        if t == -1:
                            t = pieces[p][i]
                        elif pieces[p][i] != t:
                            state1_view[i][j][4] = 1
                            break
                    else:
                        complete = False
                else:
                    if t != -1:
                        if complete:
                            state1_view[i][j][3] = 1
                        else:
                            state1_view[i][j][(t - 1) // 3] = 1

        for i in range(19):
            if self.board_[i] != -1:
                h = pieces[self.board_[i]]
                state2_view[i][0][(h[0] - 1) // 4] = 1
                state2_view[i][1][0 if h[1] == 2 else (h[1] - 5)] = 1
                state2_view[i][2][2 if h[2] == 8 else (h[2] - 3)] = 1
                state3_view[self.board_[i]] = 1

    def encode(self):
        state = np.zeros(state_size, dtype=np.float32)
        self.encode_(state)
        return torch.from_numpy(state).unsqueeze_(0)

    def __getstate__(self):
        return self.board_, self.subset_, self.step_, self.last_positions_

    def __setstate__(self, state):
        board, subset, self.step_, last_positions = state
        for i in range(19):
            self.board_[i] = board[i]
            self.last_positions_[i] = last_positions[i]
        for i in range(27):
            self.subset_[i] = subset[i]
