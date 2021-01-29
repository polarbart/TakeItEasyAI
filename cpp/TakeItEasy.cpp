#include "TakeItEasy.h"
#include <random>

const std::int8_t numbers_on_pieces[28][3] = {{1, 2, 3},
                                              {1, 2, 4},
                                              {1, 2, 8},
                                              {1, 6, 3},
                                              {1, 6, 4},
                                              {1, 6, 8},
                                              {1, 7, 3},
                                              {1, 7, 4},
                                              {1, 7, 8},
                                              {5, 2, 3},
                                              {5, 2, 4},
                                              {5, 2, 8},
                                              {5, 6, 3},
                                              {5, 6, 4},
                                              {5, 6, 8},
                                              {5, 7, 3},
                                              {5, 7, 4},
                                              {5, 7, 8},
                                              {9, 2, 3},
                                              {9, 2, 4},
                                              {9, 2, 8},
                                              {9, 6, 3},
                                              {9, 6, 4},
                                              {9, 6, 8},
                                              {9, 7, 3},
                                              {9, 7, 4},
                                              {9, 7, 8},
                                              {-1, -1, -1}};

const std::int8_t tiles_on_lines[3][5][5] = {{{0, 1,  2,  INVALID_TILE, INVALID_TILE}, {3, 4, 5,  6,  INVALID_TILE}, {7, 8, 9, 10, 11}, {12, 13, 14, 15, INVALID_TILE}, {16, 17, 18, INVALID_TILE, INVALID_TILE}},
                                             {{0, 3,  7,  INVALID_TILE, INVALID_TILE}, {1, 4, 8,  12, INVALID_TILE}, {2, 5, 9, 13, 16}, {6,  10, 14, 17, INVALID_TILE}, {11, 15, 18, INVALID_TILE, INVALID_TILE}},
                                             {{7, 12, 16, INVALID_TILE, INVALID_TILE}, {3, 8, 13, 17, INVALID_TILE}, {0, 4, 9, 14, 18}, {1,  5,  10, 15, INVALID_TILE}, {2,  6,  11, INVALID_TILE, INVALID_TILE}}};

const std::int8_t lines_on_tile[19][3] = {{0, 0, 2},
                                          {0, 1, 3},
                                          {0, 2, 4},
                                          {1, 0, 1},
                                          {1, 1, 2},
                                          {1, 2, 3},
                                          {1, 3, 4},
                                          {2, 0, 0},
                                          {2, 1, 1},
                                          {2, 2, 2},
                                          {2, 3, 3},
                                          {2, 4, 4},
                                          {3, 1, 0},
                                          {3, 2, 1},
                                          {3, 3, 2},
                                          {3, 4, 3},
                                          {4, 2, 0},
                                          {4, 3, 1},
                                          {4, 4, 2}};

const std::int8_t numbers_for_dirs[3][3] = {{1, 5, 9},
                                            {2, 6, 7},
                                            {3, 4, 8}};

TakeItEasy::TakeItEasy(const std::optional<std::int32_t> seed) {
    if (seed.has_value())
        rng = std::mt19937(seed.value());
    else
        rng = std::mt19937(std::random_device()());

    for (std::int8_t i = 0; i < 27; ++i)
        subset[i] = i;

    board[INVALID_TILE] = INVALID_PIECE;
    reset();
}

std::int8_t TakeItEasy::getRandInt(std::int8_t m) {
    return std::uniform_int_distribution<std::int32_t>(0, m - 1)(rng);
}

void TakeItEasy::reset() {
    for (std::int8_t i = 0; i < 19; ++i) {
        board[i] = INVALID_PIECE;
        last_positions[i] = INVALID_TILE;

        std::uint8_t r = getRandInt(27 - i);
        std::swap(subset[i], subset[i + r]);
    }
    step = 0;
}

void TakeItEasy::setNextPiece(const std::int8_t piece) {
    for (std::int8_t i = 0; i < 27; ++i)
        if (subset[i] == piece) {
            if (i < step)
                throw std::runtime_error("piece was already played");
            swapCurrentPieceWith(i);
            return;
        }
    throw std::runtime_error("invalid piece");
}

void TakeItEasy::swapCurrentPieceWith(const std::int8_t swp) {
    std::swap(subset[step], subset[swp]);
}

void TakeItEasy::place(const std::int8_t pos) {
    last_positions[step] = pos;
    board[pos] = subset[step];
    step++;
}

void TakeItEasy::undo() {
    step--;
    board[last_positions[step]] = INVALID_PIECE;
}

std::int16_t TakeItEasy::computeScore() const {
    // trying to avoid branches
    std::int16_t score = 0; // max score is 307
    for (std::int8_t i = 0; i < 3; ++i) // direction i. e. |, \, or /
        for (std::int8_t j = 0; j < 5; ++j) { // row
            std::int8_t reference = numbers_on_pieces[board[tiles_on_lines[i][j][0]]][i];
            std::int8_t number_of_tiles = 5 - abs(2 - j);
            bool allEqual = board[tiles_on_lines[i][j][0]] != INVALID_PIECE;
            for (std::int8_t k = 0; k < number_of_tiles; ++k) // tile
                // numbers_on_pieces[INVALID_PIECE][i] == -1
                allEqual &= (numbers_on_pieces[board[tiles_on_lines[i][j][k]]][i] == reference);
            score += allEqual * (number_of_tiles * reference);
        }
    return score;
}

std::int8_t TakeItEasy::computeScoreDelta(const std::int8_t pos) const {
    // trying to avoid branches
    std::int8_t scoreChange = 0; // max score change is (9+8+7)*5 = 120
    for (std::int8_t i = 0; i < 3; ++i) { // direction i. e. |, \, or /
        std::int8_t j = lines_on_tile[pos][i]; // row
        std::int8_t reference = numbers_on_pieces[board[pos]][i];
        bool allEqual = board[pos] != INVALID_PIECE;
        for (std::int8_t k = 0; k < 5; ++k) // tile
            allEqual &= (tiles_on_lines[i][j][k] == INVALID_TILE)
                        | (numbers_on_pieces[board[tiles_on_lines[i][j][k]]][i] == reference);
        std::int8_t number_of_tiles = 5 - abs(2 - j);
        scoreChange += allEqual * (number_of_tiles * reference);
    }
    return scoreChange;
}

void TakeItEasy::encode(std::int8_t* buf) const {
    std::int8_t (&state2)[19][3][3]  = *reinterpret_cast<std::int8_t(*)[19][3][3]>(buf);

    for (std::int8_t i = 0; i < 19; ++i)
        for (std::int8_t j = 0; j < 3; ++j)
            for (std::int8_t k = 0; k < 3; ++k)
                state2[i][j][k] = numbers_on_pieces[board[i]][j] == numbers_for_dirs[j][k];
}

