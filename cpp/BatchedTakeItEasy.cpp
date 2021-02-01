#include "BatchedTakeItEasy.h"

template <typename T, std::int8_t size>
py::array_t<T, py::array::c_style> toNumpy(T* ptr, const std::int32_t(&shape)[size]) {
    std::int32_t strides[size];
    strides[size - 1] = sizeof(T);
    for (std::int8_t i = size - 2; i >= 0; --i)
        strides[i] = shape[i+1] * strides[i+1];

    return py::array_t<T, py::array::c_style>(
            shape,
            strides,
            ptr,
            py::capsule(ptr, [](void *d) {delete[] reinterpret_cast<T*>(d);})
    );
}

BatchedTakeItEasy::BatchedTakeItEasy(const std::int32_t nGames, const std::optional<std::int32_t> seed) : nGames(nGames) {
    std::int32_t _seed;
    if (seed.has_value())
        _seed = seed.value();
    else {
        std::mt19937 rng((std::random_device()()));
        std::uniform_int_distribution<std::int32_t> uniform(0, 1000000000);
        _seed = uniform(rng);
    }

    for (std::int32_t i = 0; i < nGames; ++i)
        games.emplace_back(std::optional(_seed + i));

    previousEncodings = new std::int8_t[nGames * STATE_SIZE];

    reset();
}

BatchedTakeItEasy::~BatchedTakeItEasy() {
    delete[] previousEncodings;
}


py::tuple BatchedTakeItEasy::computeEncodings(const bool iterOverRemainingPieces) {

    const std::int32_t step = games[0].step;
    const std::int32_t nEmptySpaces = 19 - step;
    const std::int32_t nRemainingPieces = iterOverRemainingPieces ? (27 - step) : 1;

    // prepare arrays
    // the states of the first step are irrelevant since there are no pieces on the board
    // the states1 of the last step are irrelevant since there is no expected future reward
    std::int8_t *states = nullptr;
    std::int8_t *states1 = nullptr;
    if (step > 0)
        states = new std::int8_t[nGames * STATE_SIZE];
    if (step < 19)
        states1 = new std::int8_t[nGames * nRemainingPieces * nEmptySpaces * STATE_SIZE];
    auto *rewards = new std::int8_t[nGames * nRemainingPieces * nEmptySpaces];
    auto *emptySpaces = new std::int8_t[nGames * nEmptySpaces];

    const std::int32_t states1Strides[3] = {
            nRemainingPieces * nEmptySpaces * STATE_SIZE,
            nEmptySpaces * STATE_SIZE,
            STATE_SIZE
    };
    const std::int32_t rewardsStrides[2] = {
            nRemainingPieces * nEmptySpaces,
            nEmptySpaces
    };

    // iterate over games
    #pragma omp parallel for
    for (std::int32_t i = 0; i < nGames; ++i) {

        // prevEnc contains the current state
        std::int8_t* prevEnc = &previousEncodings[i * STATE_SIZE];
        if (step > 0)
            std::copy_n(prevEnc, STATE_SIZE, &states[i * STATE_SIZE]);

        std::int8_t *cStates1 = nullptr;
        if (step < 19)
            cStates1 = &states1[i * states1Strides[0]];
        auto *cRewards = &rewards[i * rewardsStrides[0]];
        auto *cEmptySpaces = &emptySpaces[i * nEmptySpaces];

        // get the spaces that are still empty i. e. games[i].board[m] == INVALID_PIECE
        for (std::int8_t m = 0, n = 0; m < 19; ++m) {
            if (games[i].board[m] == INVALID_PIECE)
                cEmptySpaces[n++] = m;
        }

        // iterate over all remaining pieces and all empty spaces
        // save the rewards and the state encodings
        for (std::int8_t j = 0; j < nRemainingPieces; ++j) {
            games[i].swapCurrentPieceWith(step + j);
            for (std::int8_t k = 0; k < nEmptySpaces; ++k) {
                games[i].place(cEmptySpaces[k]);
                cRewards[j * rewardsStrides[1] + k] = games[i].computeScoreDelta(cEmptySpaces[k]);
                if (cStates1 != nullptr)
                    deltaEncode(i, prevEnc, &cStates1[j * states1Strides[1] + k * states1Strides[2]], cEmptySpaces[k]);
                games[i].undo();
            }
            games[i].swapCurrentPieceWith(step + j);
        }
    }

    // return all arrays

    if (states == nullptr)
        return py::make_tuple(
                nullptr,
                toNumpy(states1, {nGames, nRemainingPieces, nEmptySpaces, STATE_SIZE}),
                toNumpy(rewards, {nGames, nRemainingPieces, nEmptySpaces}),
                toNumpy(emptySpaces, {nGames, nEmptySpaces})
        );

    if (states1 == nullptr)
        return py::make_tuple(
                toNumpy(states, {nGames, STATE_SIZE}),
                nullptr,
                toNumpy(rewards, {nGames, nRemainingPieces, nEmptySpaces}),
                toNumpy(emptySpaces, {nGames, nEmptySpaces})
        );

    return py::make_tuple(
            toNumpy(states, {nGames, STATE_SIZE}),
            toNumpy(states1, {nGames, nRemainingPieces, nEmptySpaces, STATE_SIZE}),
            toNumpy(rewards, {nGames, nRemainingPieces, nEmptySpaces}),
            toNumpy(emptySpaces, {nGames, nEmptySpaces})
    );
}

void BatchedTakeItEasy::place(const NumpyByteArray &a) {
    if (a.size() != nGames)
        throw std::invalid_argument("the size of \"a\" and the number of games must be the same");

    // place the pieces and save the encoding into previousEncodings
    auto r = a.unchecked<1>();
    #pragma omp parallel for
    for (std::int32_t g = 0; g < nGames; ++g) {
        games[g].place(r(g));
        if (games[0].step < 19)
            deltaEncode(g, &previousEncodings[g * STATE_SIZE], r(g));
    }
}

void BatchedTakeItEasy::reset() {

    // reset all games
    #pragma omp parallel for
    for (std::int32_t g = 0; g < nGames; ++g)
        games[g].reset();

    // initial encoding
    std::fill_n(previousEncodings, nGames * STATE_SIZE, 0);
}

NumpyWordArray BatchedTakeItEasy::computeScores() const {
    auto *scores = new std::int16_t[nGames];

    #pragma omp parallel for
    for (std::int32_t i = 0; i < nGames; ++i)
        scores[i] = games[i].computeScore();

    return toNumpy(scores, {nGames});
}

void BatchedTakeItEasy::deltaEncode(std::int32_t g, std::int8_t *src, std::int8_t *dst, std::int8_t space) const {
    std::copy_n(src, STATE_SIZE, dst);
    deltaEncode(g, dst, space);
}

void BatchedTakeItEasy::deltaEncode(const std::int32_t g, std::int8_t* encoding, const std::int8_t space) const {
    // encode the state if a piece of the space "space" was placed
    std::int8_t (&state)[19][3][3]  = *reinterpret_cast<std::int8_t(*)[19][3][3]>(encoding);
    for (std::int8_t j = 0; j < 3; ++j)
        for (std::int8_t k = 0; k < 3; ++k)
            state[space][j][k] = (numbers_on_pieces[games[g].board[space]][j] == numbers_for_dirs[j][k]);
}
