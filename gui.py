from tkinter import *
import numpy as np
import torch
from network import Network
from takeiteasy import pieces as piecesnd
import matplotlib.pyplot as plt
import os
from os.path import join
from datetime import datetime

pieces = piecesnd.tolist()
try:
    from cpp.bin.TakeItEasyC import TakeItEasy
except ImportError:
    print('Using the TakeItEasy python implementation!')
    from takeiteasy import TakeItEasy


class GameLogger:
    def __init__(self, path):
        assert path is not None
        os.makedirs(path, exist_ok=True)
        self.path = path
        self.file_name = None
        self.data = None
        self.reset()

    def save(self):
        with open(self.file_name, 'w+') as f:
            for d in self.data:
                f.write(', '.join(map(str, d)) + '\n')

    def place(self, hexagon, pos):
        """
        The spaces are numbered from top to bottom, from left to right
            3  7 12
         0  4  8 13 16
         1  5  9 14 17
         2  6 10 15 18
              11
        """

        self.data.append((hexagon, pos))
        self.save()

    def undo(self):
        del self.data[-1]
        self.save()

    def reset(self):
        self.file_name = join(self.path, f'{len(os.listdir(self.path))}_'
                                         f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.tie')
        self.data = []


class Hexagon:

    line_colors = ['gray50',
                   'LightPink1',
                   'maroon3',
                   'DodgerBlue2',
                   'SteelBlue4',
                   'red2',
                   'chartreuse3',
                   'dark orange',
                   'yellow2']

    triangles = [
        [0, 1, 2],
        [2, 3, 4],
        [4, 5, 0],
        [0, 2, 4]
    ]

    @staticmethod
    def _compute_edges(cx, cy, radius):
        hexagon_edges = []
        for i in range(6):
            x1 = np.cos(i / 6 * 2 * np.pi) * radius + cx
            y1 = np.sin(i / 6 * 2 * np.pi) * radius + cy
            hexagon_edges.append((x1, y1))
        return hexagon_edges
    
    @staticmethod
    def _compute_line_vectors(radius):
        line_vectors = []
        for i in range(3):
            x1 = .5 * (np.cos((5 - i*2) / 6 * 2 * np.pi) + np.cos((4 - i*2) / 6 * 2 * np.pi)) * radius
            x2 = .5 * (np.cos((2 - i*2) / 6 * 2 * np.pi) + np.cos((1 - i*2) / 6 * 2 * np.pi)) * radius
            y1 = .5 * (np.sin((5 - i*2) / 6 * 2 * np.pi) + np.sin((4 - i*2) / 6 * 2 * np.pi)) * radius
            y2 = .5 * (np.sin((2 - i*2) / 6 * 2 * np.pi) + np.sin((1 - i*2) / 6 * 2 * np.pi)) * radius
            line_vectors.append(((x1, y1), (x2, y2)))
        return line_vectors
    
    def __init__(self, cx, cy, radius):
        self.p_center = cx, cy
        self.radius = radius
        self.piece = None
        self.expected_value = 0
        self.best = False
        self.invisible = False
        self.p_edges = Hexagon._compute_edges(cx, cy, radius)
        self.v_lines = Hexagon._compute_line_vectors(radius)

    def draw(self, canvas, fontsize, draw_expected_values):
        if self.invisible:
            return
        cx, cy = self.p_center
        if self.piece is not None:
            for l, ((x1, y1), (x2, y2)) in zip(self.piece, self.v_lines):
                canvas.create_line(x1 + cx, y1 + cy, x2 + cx, y2 + cy, fill=self.line_colors[l - 1], width=max(round(fontsize/2), 1))
                canvas.create_text(x1 * .7 + cx, y1 * .7 + cy, fill='black', font=f'Consolas {fontsize + round(self.radius / 7)} italic bold', text=str(l))
        elif draw_expected_values:
            color = 'red' if self.best else 'black'
            canvas.create_text(cx, cy, fill=color, font=f'Consolas {fontsize} italic bold', text=f'{self.expected_value:.2f}')

        for i in range(6):
            x1, y1 = self.p_edges[i - 1]
            x2, y2 = self.p_edges[i]
            canvas.create_line(x1, y1, x2, y2, fill='black', width=max(round(fontsize/10), 1))

    @staticmethod
    def _is_point_in_triangle(p, p1, p2, p3):
        p, p1, p2, p3 = map(np.array, (p, p1, p2, p3))
        s, r = np.matmul((p - p1), np.linalg.inv(np.vstack([p2-p1, p3-p1])))
        return 0 <= s <= 1 and 0 <= r <= 1 and 0 <= s + r <= 1

    def is_point_in_hexagon(self, px, py):
        if self.invisible:
            return False
        for i, j, k in Hexagon.triangles:
            p1 = self.p_edges[i]
            p2 = self.p_edges[j]
            p3 = self.p_edges[k]
            if Hexagon._is_point_in_triangle((px, py), p1, p2, p3):
                return True
        return False


class TakeItEasyGui:

    remaining_pieces_order = [3, 1, 4, 14, 9, 13, 16, 20, 21, 25, 0, 5, 6, 7, 11, 12, 17, 15, 10, 19, 23, 24, 26, 18, 22, 8, 2]

    def _create_board_and_pieces(self, radius):

        base_x, base_y = 1.5 * radius, 4.5 * np.sqrt(3) / 2 * radius

        board = []
        tx, ty = 0, 0
        for k in range(5):
            if k > 0:
                tx += 1.5
                if k <= 2:
                    ty -= .5 * np.sqrt(3)
                else:
                    ty += .5 * np.sqrt(3)

            for i in range(5 - abs(2 - k)):
                cx = base_x + radius * tx
                cy = base_y + (ty + i * np.sqrt(3)) * radius
                board.append(Hexagon(cx, cy, radius))

        selected_piece = Hexagon(base_x + 8.5 * radius, base_y + np.sqrt(3) * radius, radius)

        cx = base_x + 9.5 * radius
        cy = base_y - np.sqrt(3) / 2 * radius

        all_pieces = [None] * 27
        s = 0
        for n in [3, 4, 3, 4, 5, 6, 2]:
            cy += (-1 if n > 3 else 1) * np.sqrt(3) / 2 * radius
            cx += 1.5 * radius
            for i in range(n):
                p = self.remaining_pieces_order[s]
                py = (cy + i * np.sqrt(3) * radius) if s < 25 else (cy + (2 * i + 1) * np.sqrt(3) * radius)
                all_pieces[p] = Hexagon(cx, py, radius)
                all_pieces[p].piece = pieces[p]
                s += 1

        return board, selected_piece, all_pieces

    def __init__(self, game, net, radius=70, game_logger_path=None):
        self.radius = radius
        self.fontsize = round(radius / 70 * 20)
        self.game = game
        self.state_size = 19*3*3
        self.game_logger = GameLogger(game_logger_path) if game_logger_path is not None else None
        self.root = Tk()
        self.root.configure(background='white')
        self.canvas = Canvas(self.root, width=23*self.radius, height=13*np.sqrt(3)/2*self.radius)
        self.canvas.configure(background='white')
        self.canvas.bind("<Button-1>", self.click)
        self.canvas.bind("<Button-3>", self.click3)
        self.canvas.pack()
        self.draw_expected_values = BooleanVar(value=True)
        self.draw_expected_values.trace('w', lambda *_: self.render())
        Checkbutton(self.root, text='show expected values', variable=self.draw_expected_values, bg='white').pack(side=LEFT)
        self.bplace = Button(self.root, text='place', command=self.place)
        self.bplace.pack(side=RIGHT)
        Button(self.root, text='undo', command=self.undo).pack(side=RIGHT)
        Button(self.root, text='reset', command=self.reset).pack(side=RIGHT)
        self.board, self.selected_piece, self.all_pieces = self._create_board_and_pieces(self.radius)
        self.net = net
        self.value_distributions = {}
        self.next_action = -1
        self.reset()

    def undo(self):
        if self.game.step == 0:
            return
        self.game.undo()
        self.compute_expected_values()
        self.render()
        self.bplace.configure(state='normal')
        if self.game_logger is not None:
            self.game_logger.undo()

    def click(self, e):
        for i, h in enumerate(self.board):
            if h.is_point_in_hexagon(e.x, e.y):
                self.place(i)
                break
        else:
            for h in self.all_pieces:
                if h.is_point_in_hexagon(e.x, e.y):
                    self.game.set_next_piece(pieces.index(h.piece))
                    self.compute_expected_values()
                    self.render()

    def click3(self, e):
        for i, h in enumerate(self.board):
            if i in self.value_distributions and h.is_point_in_hexagon(e.x, e.y):
                plt.figure()
                plt.ion()
                d, r = self.value_distributions[i]
                d = (self.game.compute_score() + r + d).cpu().numpy()
                plt.scatter(np.sort(d), ((2 * np.arange(d.shape[0]) + 1) / (2 * d.shape[0])))
                #y = 1 / ((d[1:] - d[:-1]) * d.shape[0])
                #x = (d[1:] + d[:-1]) / 2
                #plt.plot(x, y)
                plt.show()
                plt.pause(0.001)
                break

    def compute_expected_values(self):
        empty_spaces = self.game.empty_spaces
        states = torch.empty((len(empty_spaces), self.state_size), dtype=torch.float)
        rewards = torch.empty((len(empty_spaces),), dtype=torch.float)
        for i, a in enumerate(empty_spaces):
            rewards[i] = self.game.place(a)
            states[i] = torch.from_numpy(self.game.encode())
            self.game.undo()
        with torch.no_grad():
            qd = self.net(states)
            if self.game.step == 18:
                qd = torch.zeros_like(qd)
            q = rewards + qd.mean(1)

        self.value_distributions.clear()
        for i, d, r in zip(empty_spaces, qd, rewards):
            self.value_distributions[i] = d, r

        self.next_action = empty_spaces[q.argmax()]
        value = self.game.compute_score()
        for i, e in zip(empty_spaces, q):
            self.board[i].expected_value = float(e) + value

    def place(self, action=None):
        if action is not None and action not in self.game.empty_spaces:
            return
        if action is None:
            action = self.next_action
        if self.game_logger is not None:
            self.game_logger.place(action, self.game.next_piece)
        self.game.place(action)
        if self.game.step == 19:
            self.bplace.configure(state='disabled')
        else:
            self.compute_expected_values()
        self.render()

    def reset(self):
        self.game.reset()
        self.compute_expected_values()
        self.bplace.configure(state='normal')
        self.render()
        if self.game_logger is not None:
            self.game_logger.reset()

    def render(self):
        self.canvas.delete('all')
        # print('rendering')

        for i, h in enumerate(self.board):
            h.piece = self.game.get_piece_at(i)
            h.best = i == self.next_action
            h.draw(self.canvas, self.fontsize, self.draw_expected_values.get())

        if self.game.step < 19:
            self.selected_piece.piece = self.game.next_piece
            self.selected_piece.draw(self.canvas, self.fontsize, self.draw_expected_values.get())
        cx, cy = self.selected_piece.p_center
        self.canvas.create_text(cx, cy + 1.5 * self.radius, fill='black', font=f'Consolas {self.fontsize} italic bold', text=f'score: {self.game.compute_score()}')

        remaining_pieces = self.game.remaining_pieces
        for h in self.all_pieces:
            h.invisible = h.piece not in remaining_pieces or h.piece == self.selected_piece.piece
            h.draw(self.canvas, self.fontsize, self.draw_expected_values.get())

        self.root.update()


if __name__ == '__main__':
    net = Network(19*3*3, 100, 2048)
    net.load_state_dict(torch.load('qr_network', map_location='cpu'))
    r = TakeItEasyGui(
        TakeItEasy(seed=None),
        net,
        radius=70,
        game_logger_path='game_logs'
    )
    mainloop()
