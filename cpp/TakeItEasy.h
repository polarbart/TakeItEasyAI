//
// Created by Julius on 17.05.2020.
//

#ifndef CPP_TAKEITEASY_H
#define CPP_TAKEITEASY_H

#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#define STATE_SIZE (3*5*5 + 19*3*3 + 27)
#define INVALID_TILE 19
#define INVALID_PIECE 27

namespace py = pybind11;

typedef py::array_t<std::float_t , py::array::c_style> NumpyFloatArray;
typedef py::array_t<std::int16_t , py::array::c_style> NumpyWordArray;
typedef py::array_t<std::int8_t , py::array::c_style> NumpyByteArray;

extern const std::int8_t numbers_on_pieces[28][3];
extern const std::int8_t tiles_on_lines[3][5][5];
extern const std::int8_t lines_on_tile[19][3];
extern const std::int8_t numbers_for_dirs[3][3];

class TakeItEasy {

public:
    explicit TakeItEasy(std::int32_t seed = -1);
    void reset();
    void setNextPiece(std::int8_t piece);
    void swapCurrentPieceWith(std::int8_t swp);
    std::int8_t place(std::int8_t pos);
    void undo();

    [[nodiscard]] std::int16_t computeScore() const; // maximal score change is 307 > 255
    [[nodiscard]] std::int8_t computeScoreDelta(std::int8_t pos) const; // maximal score change is (9+8+7)*5 = 120 < 127

    void encode(std::float_t* buf) const;
private:
    void setEncoding(const std::float_t *buf, std::int8_t s);
    void deltaEncode(std::float_t* buf, std::int8_t pos);

public:
    // The round in which the game is. Also equals the number of pieces already on the board. Therefore between 0 and 19
    std::int8_t step;
    std::int8_t board[20]; // + invalid pos
    std::int8_t subset[27];
    std::int8_t last_positions[19];
    std::float_t encodings[19][STATE_SIZE];

public:
    class BatchedTakeItEasy {

    public:
        explicit BatchedTakeItEasy(std::int32_t nGames, std::int32_t seed = -1);
        void place(const py::array_t<std::int8_t, py::array::c_style> &a);
        void reset();
        [[nodiscard]] NumpyWordArray computeScores() const;
        py::tuple computeEncodings(bool iterOverRemainingPieces);

        std::vector<TakeItEasy> games;
    };
};


#endif //CPP_TAKEITEASY_H
