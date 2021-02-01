#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <optional>
#include "TakeItEasy.h"
#include "BatchedTakeItEasy.h"

namespace py = pybind11;

typedef std::array<std::int8_t, 3> Piece;

std::vector<std::uint8_t> empty_spaces(const TakeItEasy &t) {
    std::vector<std::uint8_t> ret;
    for (std::int8_t i = 0; i < 19; ++i)
        if (t.board[i] == INVALID_PIECE)
            ret.push_back(i);
    return ret;
}

std::vector<Piece> remaining_pieces(const TakeItEasy &t) {
    std::vector<Piece> ret;
    for (std::int8_t i = t.step; i < 27; ++i)
        ret.push_back({
                              numbers_on_pieces[t.subset[i]][0],
                              numbers_on_pieces[t.subset[i]][1],
                              numbers_on_pieces[t.subset[i]][2]
                      }
        );
    return ret;
}

std::optional<Piece> get_piece_at(const TakeItEasy &t, std::int8_t i) {
    if (t.board[i] == INVALID_PIECE)
        return std::nullopt;
    return std::optional<Piece>({
                                        numbers_on_pieces[t.board[i]][0],
                                        numbers_on_pieces[t.board[i]][1],
                                        numbers_on_pieces[t.board[i]][2]
                                });
}

std::optional<Piece> next_piece(const TakeItEasy &t) {
    if (t.step >= 19)
        return std::nullopt;
    return std::optional<Piece>({
                                        numbers_on_pieces[t.subset[t.step]][0],
                                        numbers_on_pieces[t.subset[t.step]][1],
                                        numbers_on_pieces[t.subset[t.step]][2]
                                });
}

py::array_t<std::int8_t, py::array::c_style> encode(const TakeItEasy &t) {

    auto *data = new std::int8_t[STATE_SIZE];
    t.encode(data);

    return py::array_t<std::int8_t, py::array::c_style>(
            {STATE_SIZE},
            {sizeof(std::int8_t)},
            data,
            py::capsule (data, [](void *d) { delete[] reinterpret_cast<std::int8_t *>(d); })
    );
}

std::int8_t place(TakeItEasy &t, const std::int8_t pos) {
    t.place(pos);
    return t.computeScoreDelta(pos);
}


PYBIND11_MODULE(TakeItEasyC, m) {
    auto tie = py::class_<TakeItEasy>(m, "TakeItEasy");
    tie.def(py::init<std::optional<std::int32_t>>(), py::arg("seed") = std::nullopt);
    tie.def("reset", &TakeItEasy::reset);
    tie.def("place", &place);
    tie.def("undo", &TakeItEasy::undo);
    tie.def("compute_score", &TakeItEasy::computeScore);
    tie.def("encode", &encode);
    tie.def("set_next_piece", &TakeItEasy::setNextPiece);
    tie.def("swap_current_piece_with", &TakeItEasy::swapCurrentPieceWith);

    tie.def_readonly("step", &TakeItEasy::step);
    tie.def_property_readonly("empty_spaces", &empty_spaces);
    tie.def_property_readonly("remaining_pieces", &remaining_pieces);
    tie.def("get_piece_at", &get_piece_at);
    tie.def_property_readonly("next_piece", &next_piece);

    auto btie = py::class_<BatchedTakeItEasy, std::shared_ptr<BatchedTakeItEasy>>(m, "BatchedTakeItEasy");
    btie.def(py::init<std::int32_t, std::optional<std::int32_t>>(), py::arg("nGames"), py::arg("seed") = std::nullopt);
    btie.def("place", &BatchedTakeItEasy::place);
    btie.def("reset", &BatchedTakeItEasy::reset);
    btie.def("compute_encodings", &BatchedTakeItEasy::computeEncodings, py::arg("iterOverRemainingPieces") = true);
    btie.def("compute_scores", &BatchedTakeItEasy::computeScores);

    btie.def(py::pickle(
            [](const BatchedTakeItEasy &t) {
                return py::make_tuple(t.nGames);
            },
            [](const py::tuple &s) {
                return std::make_shared<BatchedTakeItEasy>(s[0].cast<std::int32_t>(), std::nullopt);
            }
    ));
}



