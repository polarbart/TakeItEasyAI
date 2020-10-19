//
// Created by Julius on 17.05.2020.
//

#include "TakeItEasy.h"
#include <random>
#include <ctime>

// Every piece is uniquely identified by the numbers on it (counter clockwise)
// Every piece also has an id. This array maps from id to the numbers on the pieces
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

TakeItEasy::TakeItEasy(const std::int32_t seed) {

    if (seed == -1)
        srand(time(NULL));
    else
        srand(seed);

    for (std::int8_t i = 0; i < 27; ++i)
        subset[i] = i;

    board[INVALID_TILE] = INVALID_PIECE;
    std::fill(&encodings[0][0], &encodings[0][0] + sizeof(encodings) / sizeof(std::float_t), 0);
    reset();
}

void TakeItEasy::reset() {
    for (std::int8_t i = 0; i < 19; ++i) {
        board[i] = INVALID_PIECE;
        last_positions[i] = INVALID_TILE;

        std::uint8_t r = rand() % (27 - i);
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

std::int8_t TakeItEasy::place(const std::int8_t pos) {
    last_positions[step] = pos;
    board[pos] = subset[step];
    step++;
    return computeScoreDelta(pos);
}

void TakeItEasy::undo() {
    step--;
    board[last_positions[step]] = INVALID_PIECE;
}

std::int16_t TakeItEasy::computeScore() const {
    std::int16_t score = 0;
    for (std::int8_t i = 0; i < 3; ++i)
        for (std::int8_t j = 0; j < 5; ++j) {
            std::int8_t reference = numbers_on_pieces[board[tiles_on_lines[i][j][0]]][i];
            bool allEqual = board[tiles_on_lines[i][j][0]] != INVALID_PIECE;
            for (std::int8_t k = 0; k < 5; ++k)
                allEqual &= (tiles_on_lines[i][j][k] == INVALID_TILE)
                            | (numbers_on_pieces[board[tiles_on_lines[i][j][k]]][i] == reference);
            std::int8_t number_of_tiles = 5 - abs(2 - j);
            score += allEqual * (number_of_tiles * reference);
        }
    return score;
}

std::int8_t TakeItEasy::computeScoreDelta(const std::int8_t pos) const {
    // there should be no branch
    std::int8_t scoreChange = 0;
    for (std::int8_t i = 0; i < 3; ++i) {
        std::int8_t j = lines_on_tile[pos][i];
        std::int8_t reference = numbers_on_pieces[board[pos]][i];
        bool allEqual = true;
        for (std::int8_t k = 0; k < 5; ++k)
            allEqual &= (tiles_on_lines[i][j][k] == INVALID_TILE)
                        | ((board[tiles_on_lines[i][j][k]] != INVALID_PIECE)
                           & (numbers_on_pieces[board[tiles_on_lines[i][j][k]]][i] == reference));
        std::int8_t number_of_tiles = 5 - abs(2 - j);
        scoreChange += allEqual * (number_of_tiles * reference);
    }
    return scoreChange;
}

void TakeItEasy::encode(std::float_t* buf) const {
    std::float_t (&state1)[3][5][5]  = *reinterpret_cast<std::float_t(*)[3][5][5]>(buf);
    std::float_t (&state2)[19][3][3]  = *reinterpret_cast<std::float_t(*)[19][3][3]>(&buf[3*5*5]);
    std::float_t (&state3)[27]  = *reinterpret_cast<std::float_t(*)[27]>(&buf[3*5*5 + 19*3*3]);

    for (std::int8_t i = 0; i < 3; ++i)
        for (std::int8_t j = 0; j < 5; ++j) {
            bool encoding[5] = {true, true, true, true, false};
            for (std::int8_t k = 0; k < 5; ++k) {
                for (std::int8_t m = 0; m < 3; ++m)
                    encoding[m] &= (board[tiles_on_lines[i][j][k]] == INVALID_PIECE)
                                 | (numbers_for_dirs[i][m] == numbers_on_pieces[board[tiles_on_lines[i][j][k]]][i]);
                encoding[3] &= (tiles_on_lines[i][j][k] == INVALID_TILE)
                               | (board[tiles_on_lines[i][j][k]] != INVALID_PIECE);
                encoding[4] |= (!(tiles_on_lines[i][j][k] == INVALID_TILE))
                               & (board[tiles_on_lines[i][j][k]] != INVALID_PIECE);
            }
            for (std::int8_t m = 0; m < 3; ++m)
                encoding[m] &= encoding[4];
            encoding[3] &= encoding[0] | encoding[1] | encoding[2];
            for (std::int8_t m = 0; m < 3; ++m)
                encoding[m] &= !encoding[3];
            encoding[4] &= !(encoding[0] | encoding[1] | encoding[2] | encoding[3]);
            for (std::int8_t m = 0; m < 5; ++m)
                state1[i][j][m] = 1. * encoding[m];
        }

    for (std::int8_t i = 0; i < 19; ++i)
        for (std::int8_t j = 0; j < 3; ++j)
            for (std::int8_t k = 0; k < 3; ++k)
                state2[i][j][k] = 1. * (numbers_on_pieces[board[i]][j] == numbers_for_dirs[j][k]);

    for (std::int8_t i = 0; i < 27; ++i)
        state3[subset[i]] = 1. * (i < step);

}

void TakeItEasy::deltaEncode(std::float_t *buf, const std::int8_t pos) {
    for (std::int16_t i = 0; i < STATE_SIZE; ++i)
        buf[i] = encodings[step - 1][i];

    std::float_t (&state1)[3][5][5]  = *reinterpret_cast<std::float_t(*)[3][5][5]>(buf);
    std::float_t (&state2)[19][3][3]  = *reinterpret_cast<std::float_t(*)[19][3][3]>(&buf[3*5*5]);
    std::float_t (&state3)[19]  = *reinterpret_cast<std::float_t(*)[19]>(&buf[3*5*5 + 19*3*3]);

    for (std::int8_t i = 0; i < 3; ++i) {
        std::int8_t j = lines_on_tile[pos][i];
        bool encoding[5] = {true, true, true, true, false};
        for (std::int8_t k = 0; k < 5; ++k) {
            for (std::int8_t m = 0; m < 3; ++m)
                encoding[m] &= (board[tiles_on_lines[i][j][k]] == INVALID_PIECE)
                               | (numbers_for_dirs[i][m] == numbers_on_pieces[board[tiles_on_lines[i][j][k]]][i]);
            bool has_valid_piece = (tiles_on_lines[i][j][k] == INVALID_TILE)
                                   | (board[tiles_on_lines[i][j][k]] != INVALID_PIECE);
            encoding[3] &= has_valid_piece;
            encoding[4] |= has_valid_piece;
        }
        encoding[3] &= encoding[0] | encoding[1] | encoding[2];
        for (std::int8_t m = 0; m < 3; ++m)
            encoding[m] &= !encoding[3];
        encoding[4] &= !(encoding[0] | encoding[1] | encoding[2] | encoding[3]);
        for (std::int8_t m = 0; m < 5; ++m)
            state1[i][j][m] = 1. * encoding[m];
    }

    for (std::int8_t j = 0; j < 3; ++j)
        for (std::int8_t k = 0; k < 3; ++k)
            state2[pos][j][k] = 1. * (numbers_on_pieces[board[pos]][j] == numbers_for_dirs[j][k]);

    state3[board[pos]] = 1.;
}

void TakeItEasy::setEncoding(const std::float_t *buf, const std::int8_t s) {
    for (std::int16_t i = 0; i < STATE_SIZE; ++i)
        encodings[s][i] = buf[i];
}

TakeItEasy::BatchedTakeItEasy::BatchedTakeItEasy(std::int32_t nGames, std::int32_t seed) {

    for (std::int32_t i = 0; i < nGames; ++i)
        games.emplace_back(0);

    if (seed == -1)
        seed = time(nullptr);

    srand(seed);

    for (std::int32_t i = 0; i < nGames; ++i)
        games[i].encode(games[i].encodings[0]);
}

py::tuple TakeItEasy::BatchedTakeItEasy::computeEncodings(const bool iterOverRemainingPieces) {

    const std::int32_t nGames = games.size();
    const std::int32_t step = games[0].step;
    const std::int32_t nEmptyTiles = 19 - step;
    const std::int32_t nRemainingPieces = iterOverRemainingPieces ? (27 - step) : 1;

    std::float_t *states = nullptr;
    if (step > 0)
        states = new std::float_t[nGames * STATE_SIZE];
    auto *states1 = new std::float_t[nGames * nRemainingPieces * nEmptyTiles * STATE_SIZE];
    auto *rewards = new std::float_t[nGames * nRemainingPieces * nEmptyTiles];
    auto *emptyTiles = new std::int8_t[nGames * nEmptyTiles];

    const std::int32_t states1Strides[3] = {
            nRemainingPieces * nEmptyTiles * STATE_SIZE,
            nEmptyTiles * STATE_SIZE,
            STATE_SIZE
    };
    const std::int32_t rewardsStrides[2] = {
            nRemainingPieces * nEmptyTiles,
            nEmptyTiles
    };

    #pragma omp parallel for
    for (std::int32_t i = 0; i < nGames; ++i) {

        if (step > 0) {
            std::float_t *buf = &states[i * STATE_SIZE];
            games[i].deltaEncode(buf, games[i].last_positions[step - 1]);
            games[i].setEncoding(buf, step);
        }

        auto *cStates1 = &states1[i * states1Strides[0]];
        auto *cRewards = &rewards[i * rewardsStrides[0]];
        auto *cEmptyTiles = &emptyTiles[i * nEmptyTiles];

        for (std::int8_t m = 0, n = 0; m < 19; ++m) {
            if (games[i].board[m] == INVALID_PIECE)
                cEmptyTiles[n++] = m;
        }

        for (std::int8_t j = 0; j < nRemainingPieces; ++j) {
            games[i].swapCurrentPieceWith(step + j);
            for (std::int8_t k = 0; k < nEmptyTiles; ++k) {
                cRewards[j * rewardsStrides[1] + k] = games[i].place(cEmptyTiles[k]);
                games[i].deltaEncode(&cStates1[j * states1Strides[1] + k * states1Strides[2]], cEmptyTiles[k]);
                games[i].undo();
            }
            games[i].swapCurrentPieceWith(step + j);
        }
    }


    NumpyFloatArray numpyStates1(
            {nGames, nRemainingPieces, nEmptyTiles, STATE_SIZE},
            {
                    nRemainingPieces * nEmptyTiles * STATE_SIZE * sizeof(std::float_t),
                    nEmptyTiles * STATE_SIZE * sizeof(std::float_t),
                    STATE_SIZE * sizeof(std::float_t),
                    sizeof(std::float_t)
            },
            states1,
            py::capsule(states1, [](void *d) {delete[] reinterpret_cast<std::float_t*>(d);})
    );

    NumpyFloatArray numpyRewards(
            {nGames, nRemainingPieces, nEmptyTiles},
            {
                    nRemainingPieces * nEmptyTiles * sizeof(std::float_t),
                    nEmptyTiles * sizeof(std::float_t),
                    sizeof(std::float_t)
            },
            rewards,
            py::capsule(rewards, [](void *d) {delete[] reinterpret_cast<std::float_t*>(d);})
    );

    NumpyByteArray numpyEmptyTiles(
            {nGames, nEmptyTiles},
            {nEmptyTiles * sizeof(std::int8_t), sizeof(std::int8_t)},
            emptyTiles,
            py::capsule(emptyTiles, [](void *d) {delete[] reinterpret_cast<std::int8_t*>(d);})
    );

    if (step == 0)
        return py::make_tuple(
                nullptr,
                numpyStates1,
                numpyRewards,
                numpyEmptyTiles
        );

    return py::make_tuple(
            NumpyFloatArray(
                    {nGames, STATE_SIZE},
                    {
                            STATE_SIZE * sizeof(std::float_t),
                            sizeof(std::float_t)
                    },
                    states,
                    py::capsule(states, [](void *d) {delete[] reinterpret_cast<std::float_t*>(d);})
            ),
            numpyStates1,
            numpyRewards,
            numpyEmptyTiles
    );
}

void TakeItEasy::BatchedTakeItEasy::place(const NumpyByteArray &a) {
    if (a.size() != games.size())
        throw std::invalid_argument("the size of \"a\" and the number of games must be the same");
    auto r = a.unchecked<1>();
    for (std::int32_t i = 0; i < games.size(); ++i)
        games[i].place(r(i));
}

void TakeItEasy::BatchedTakeItEasy::reset() {
    for (auto &g : games)
        g.reset();
}

NumpyWordArray TakeItEasy::BatchedTakeItEasy::computeScores() const {
    auto *scores = new std::int16_t[games.size()];

    for (std::int32_t i = 0; i < games.size(); ++i)
        scores[i] = games[i].computeScore();

    return NumpyWordArray(
            {games.size()},
            {sizeof(std::int16_t)},
            scores,
            py::capsule(scores, [](void *d) {delete[] reinterpret_cast<std::int16_t*>(d);})
    );
}


