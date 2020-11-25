#ifndef TAKEITEASYC_BATCHEDTAKEITEASY_H
#define TAKEITEASYC_BATCHEDTAKEITEASY_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "TakeItEasy.h"

namespace py = pybind11;

typedef py::array_t<std::float_t, py::array::c_style> NumpyFloatArray;
typedef py::array_t<std::int16_t, py::array::c_style> NumpyWordArray;
typedef py::array_t<std::int8_t, py::array::c_style> NumpyByteArray;


class BatchedTakeItEasy {

public:
    explicit BatchedTakeItEasy(std::int32_t nGames, std::optional<std::int32_t> seed);
    ~BatchedTakeItEasy();
    void place(const py::array_t<std::int8_t, py::array::c_style> &a);
    void reset();
    [[nodiscard]] NumpyWordArray computeScores() const;
    py::tuple computeEncodings(bool iterOverRemainingPieces);
    const std::int32_t nGames;

private:

    void deltaEncode(std::int32_t g, std::int8_t* encoding, std::int8_t tile) const;
    void deltaEncode(std::int32_t g, std::int8_t* src, std::int8_t* dst, std::int8_t tile) const;

    std::vector<TakeItEasy> games;
    std::int8_t* previousEncodings = nullptr;
};


#endif //TAKEITEASYC_BATCHEDTAKEITEASY_H
